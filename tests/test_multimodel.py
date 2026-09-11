"""The registry, scheduling and persistence must work with more than two models."""
import copy
import json
import time
from pathlib import Path
from threading import Event
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.registry import MODEL_REGISTRY, ModelSpec
from backend.store import TaskStore
from backend.assessment import assess
from test_workbench import FakeDetector, png


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv('ADKIT_DATA_DIR', str(tmp_path))
    for id in ['third', 'fourth']:
        monkeypatch.setitem(MODEL_REGISTRY, id, ModelSpec(id, f'Model {id}', f'TEST_{id.upper()}_WEIGHTS', 'unused.pt'))
    records = {}
    with patch('adkit.create_detector', side_effect=lambda name, **kw: FakeDetector(name, records)), patch(
            'adkit.load_detector', side_effect=lambda name, *a, **kw: FakeDetector(name, records)), TestClient(create_app()) as client:
        response = client.post('/api/tasks', json=dict(name='multi', algorithms=['anomalydino'], image_size=None))
        assert response.status_code == 201
        base = '/api/tasks/' + response.json()['id']
        for kind in ['normal', 'test']:
            assert client.post(base+'/images/'+kind, files=[('files', ('a.png', png(), 'image/png'))]).status_code == 201
        yield client, base, tmp_path, records


def run(client, base, operation, models):
    response = client.post(base+'/jobs/'+operation, json={'algorithms': models})
    assert response.status_code == 202, response.text
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        task = client.get(base).json()
        if task['job']['state'] not in ['queued', 'running']:
            return task
        time.sleep(.02)
    pytest.fail('Job did not finish')


def test_four_models_selection_addition_and_isolated_reruns(workspace):
    client, base, root, records = workspace
    models = [m['id'] for m in client.get('/api/settings').json()['algorithms']]
    assert models == ['anomalydino', 'subspacead', 'superadd', 'third', 'fourth']
    first = run(client, base, 'fit', ['anomalydino'])
    first = run(client, base, 'predict', ['anomalydino'])
    original = copy.deepcopy(first['results'])
    history = copy.deepcopy(first['models']['anomalydino']['history'])
    added = client.patch(base+'/models', json={'algorithms':['third','fourth']}).json()
    assert added['results'] == original
    assert added['models']['anomalydino']['history'] == history
    # Selection may introduce a registered model without an extra attach request.
    run(client,base,'fit',['subspacead','third','fourth'])
    task = run(client,base,'predict',models)
    assert len(task['results']) == 4 and task['job']['completed'] == 4
    assert all(m['model_ready'] for m in task['models'].values())
    assert len(list((root/task['id']).glob('model_*.pt'))) == 4
    for id in models:
        assert records[id][0].equal(records['anomalydino'][0])
        record = task['models'][id]['history'][-1]
        assert record['normal_ids'] == [task['normal'][0]['id']]
        assert record['test_ids'] == [task['test'][0]['id']]
        assert record['data_revision'] == task['data_revision']
    other_results = [r for r in task['results'] if r['algorithm'] != 'third']
    other_states = {id:task['models'][id] for id in models if id != 'third'}
    task = run(client,base,'fit',['third'])
    assert task['results'] == other_results
    assert {id:task['models'][id] for id in other_states} == other_states
    task = run(client,base,'predict',['third'])
    assert [r for r in task['results'] if r['algorithm'] != 'third'] == other_results
    assert len(task['results']) == 4  # replacement, never duplicates


def test_analyze_prepares_and_reuses_reference(workspace):
    client, base, root, _ = workspace
    task = run(client, base, 'analyze', ['anomalydino', 'third'])
    assert task['job']['state'] == 'done'
    assert len(task['results']) == 2
    checkpoint = root / task['id'] / task['models']['anomalydino']['checkpoint']
    modified = checkpoint.stat().st_mtime_ns
    task = run(client, base, 'analyze', ['anomalydino'])
    assert checkpoint.stat().st_mtime_ns == modified
    assert len(task['results']) == 2
    client.post(base + '/images/normal', files=[('files', ('b.png', png(), 'image/png'))])
    task = run(client, base, 'analyze', ['anomalydino'])
    assert task['models']['anomalydino']['fit_revision'] == task['normal_revision']
    assert task['job']['state'] == 'done'


def test_binary_threshold_and_area(workspace):
    import io
    import numpy as np
    from PIL import Image
    client, base, root, _ = workspace
    task = run(client, base, 'analyze', ['anomalydino'])
    result = task['results'][0]
    np.save(root / task['id'] / 'results' / result['raw_map'],
            np.array([[1., 1., 0., 0.], [0., 0., 0., 1.]]))
    url = base + '/binary/anomalydino/' + result['id']
    response = client.get(url, params={'threshold': 1})
    assert response.status_code == 200
    assert np.count_nonzero(np.array(Image.open(io.BytesIO(response.content)))) == 3
    response = client.get(url, params={'threshold': 1, 'area_threshold': 2})
    assert np.count_nonzero(np.array(Image.open(io.BytesIO(response.content)))) == 2
    assert client.get(url, params={'threshold': -1}).status_code == 422
    assert client.get(url, params={'threshold': 1, 'area_threshold': -1}).status_code == 422
    assert client.get(base + '/binary/anomalydino/missing', params={'threshold': 1}).status_code == 404


def test_failure_and_unfitted_model_do_not_stop_remaining_models(workspace):
    client, base, _, records = workspace
    run(client,base,'fit',['anomalydino','third'])
    with patch('adkit.load_detector', side_effect=lambda name,*a,**kw: (_ for _ in ()).throw(RuntimeError('broken model')) if name == 'anomalydino' else FakeDetector(name,records)):
        task = run(client,base,'predict',['anomalydino','fourth','third'])
    assert task['models']['anomalydino']['state'] == 'failed'
    assert task['models']['fourth']['state'] == 'failed'
    assert '请先建库' in task['models']['fourth']['message']
    assert task['models']['third']['state'] == 'done'
    assert [r['algorithm'] for r in task['results']] == ['third']
    assert task['job']['completed'] == 3
    task = run(client,base,'predict',['anomalydino'])
    assert task['models']['anomalydino']['state'] == 'done'
    assert len(task['models']['anomalydino']['history']) == 3


def test_data_revisions_invalidate_only_dependencies(workspace):
    client,base,root,_ = workspace
    models = ['anomalydino','third','fourth']
    run(client,base,'fit',models)
    task = run(client,base,'predict',models)
    revision, normal_revision = task['data_revision'], task['normal_revision']
    id = task['test'][0]['id']
    task = client.patch(base+f'/images/test/{id}/label',json={'label':'defect'}).json()
    assert task['data_revision'] == revision and len(task['results']) == 3
    # New test data clears scores but keeps each reference bank.
    task = client.post(base+'/images/test',files=[('files',('b.png',png(),'image/png'))]).json()
    assert task['data_revision'] == revision+1 and task['normal_revision'] == normal_revision
    assert not task['results'] and all(m['model_ready'] for m in task['models'].values())
    task = run(client,base,'predict',['third'])
    assert len(task['results']) == 2
    assert all(r['data_revision'] == revision+1 for r in task['results'])
    assert task['models']['fourth']['results_revision'] is None
    # Stale records cannot accidentally enter current comparison statistics.
    stale = copy.deepcopy(task); stale['results'][0]['data_revision'] = revision
    assert len(assess(stale,root/task['id'],'third',.4,0)['rows']) == 1
    task = client.post(base+'/images/normal',files=[('files',('c.png',png(),'image/png'))]).json()
    assert task['normal_revision'] == normal_revision+1 and not task['results']
    assert not any(m['model_ready'] for m in task['models'].values())
    assert not list((root/task['id']).glob('model*.pt'))
    assert task['models']['third']['history'][-1]['data_revision'] == revision+1


def test_selection_validation_and_dynamic_thresholds(workspace):
    client, base, _, _ = workspace
    for selected in [[],['missing'],['../path']]:
        assert client.post(base+'/jobs/fit',json={'algorithms':selected}).status_code in [400,422]
    assert client.post('/api/tasks',json={'name':'old special mode','algorithms':['comparison']}).status_code == 400
    assert client.patch(base+'/models',json={'algorithms':['third','third']}).json()['algorithms'] == ['anomalydino','third']
    assert client.patch(base+'/threshold',json={'algorithm':'third','threshold':.8,'area_threshold':2}).status_code == 200
    assert client.post(base+'/assessment',json={'algorithm':'third','threshold':.8}).status_code == 200
    assert client.patch(base+'/threshold',json={'algorithm':'fourth','threshold':.8}).status_code == 400


@pytest.mark.parametrize('algorithm', ['anomalydino','comparison'])
def test_legacy_migration_and_restart(tmp_path,algorithm):
    id = 'a'*32; directory = tmp_path/id; directory.mkdir()
    task = dict(id=id,algorithm=algorithm,created_at='2026-01-01',normal=[],test=[],results=[],
        threshold=.4,area_threshold=5,model_ready=True,job={'state':'done'})
    (directory/'task.json').write_text(json.dumps(task))
    (directory/'model.pt').write_text('old weights')
    store=TaskStore(tmp_path); saved=store.read(id)
    assert saved['schema_version'] == 3 and 'algorithm' not in saved
    assert len(saved['algorithms']) == (2 if algorithm == 'comparison' else 1)
    assert all(m['fit_revision'] == 0 for m in saved['models'].values())
    if algorithm != 'comparison':
        assert saved['models'][algorithm]['checkpoint'] == 'model.pt'
        assert saved['thresholds'][algorithm]['threshold'] == .4
    model=saved['models'][saved['algorithms'][0]]
    model.update(state='running',history=[{'state':'running'}])
    saved['job']['state']='running'; store.save(saved)
    restored=TaskStore(tmp_path).read(id)
    assert restored['job']['state'] == 'failed'
    assert restored['models'][saved['algorithms'][0]]['history'][-1]['state'] == 'failed'


def test_registered_adapter_uses_its_own_patch_size(workspace):
    client, base, _, records = workspace
    class OtherPatchDetector(FakeDetector):
        patch_size = 16
    with patch('adkit.create_detector',side_effect=lambda name,**kw:OtherPatchDetector(name,records)), patch(
            'adkit.load_detector',side_effect=lambda name,*a,**kw:OtherPatchDetector(name,records)):
        run(client,base,'fit',['third'])
        task=run(client,base,'predict',['third'])
    assert records['third'][0].shape[-2:] == (16,64)
    assert task['results'][0]['input_shape'] == [16,64]


def test_running_task_keeps_labels_and_thresholds_and_blocks_data_changes(workspace):
    client, base, _, records = workspace
    entered, release = Event(), Event()
    class SlowDetector(FakeDetector):
        def fit(self, batches):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('Test gate not released')
            super().fit(batches)
    try:
        with patch('adkit.create_detector',side_effect=lambda name,**kw:SlowDetector(name,records)):
            assert client.post(base+'/jobs/fit',json={'algorithms':['anomalydino','third']}).status_code == 202
            assert entered.wait(5)
            task=client.get(base).json(); id=task['test'][0]['id']
            assert client.patch(base+f'/images/test/{id}/label',json={'label':'defect'}).status_code == 200
            assert client.patch(base+'/threshold',json={'algorithm':'third','threshold':.9}).status_code == 200
            assert client.patch(base+'/models',json={'algorithms':['fourth']}).status_code == 409
            assert client.post(base+'/jobs/fit',json={'algorithms':['fourth']}).status_code == 409
            release.set()
            for _ in range(200):
                task=client.get(base).json()
                if task['job']['state'] not in ['queued','running']: break
                time.sleep(.02)
            assert task['job']['state'] == 'done'
            assert task['test'][0]['label'] == 'defect'
            assert task['thresholds']['third']['threshold'] == .9
    finally:
        release.set()
