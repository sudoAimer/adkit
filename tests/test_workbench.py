"""Decision boundaries, labels and paired execution without production weights."""
from io import BytesIO
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from backend.app import create_app
from backend.assessment import assess
from backend.engine import DetectorEngine
from backend.schemas import TaskCreate, ThresholdUpdate


def test_task_size_validation():
    for size in [None, 12, 1025, [12,29]]:
        assert TaskCreate(name='test', image_size=size)
    for size in [0, -1, [3,0], 1.5, True]:
        with pytest.raises(ValidationError):
            TaskCreate(name='test', image_size=size)
    for value in [-1, float('nan'), float('inf')]:
        with pytest.raises(ValidationError):
            ThresholdUpdate(threshold=value)


def test_confusion_and_connected_area(tmp_path):
    output = tmp_path/'results'; output.mkdir()
    # Two separated 2-pixel regions must not count as one 4-pixel defect.
    amap = np.zeros((6,6), dtype=np.float32)
    amap[0,:2] = .5; amap[4,4:] = .5
    np.save(output/'raw.npy', amap)
    task = {'algorithm':'comparison', 'test':[
        {'id':'fp','label':'normal'}, {'id':'tp','label':'defect'},
        {'id':'fn','label':'defect'}, {'id':'tn','label':'normal'},
        {'id':'unknown'}, {'id':'bad','label':'defect'}, {'id':'pending'}], 'results':[]}
    for id, score in [('fp',.5), ('tp',.5), ('fn',.49), ('tn',.49), ('unknown',.5)]:
        task['results'].append(dict(id=id, score=score, algorithm='anomalydino', raw_map='raw.npy'))
    task['results'].append(dict(id='bad', error='failed', algorithm='anomalydino'))
    task['results'].append(dict(id='fp', score=0., algorithm='subspacead', raw_map='raw.npy'))
    report = assess(task, tmp_path, 'anomalydino', .5, 2)
    assert report['counts'] == dict(false_positive=1,false_negative=1,detected=1,true_negative=1,
        unlabelled=1,unset=0,failed=1,pending=1,predicted_defect=3)
    assert report['recall'] == report['false_positive_rate'] == .5
    assert assess(task,tmp_path,'anomalydino',.5,3)['counts']['false_negative'] == 2
    assert assess(task,tmp_path,'anomalydino',None,0)['counts']['unset'] == 5
    assert assess(task,tmp_path,'subspacead',.5,0)['counts']['true_negative'] == 1
    task['results'][0].pop('raw_map')
    assert assess(task,tmp_path,'anomalydino',.5,2)['counts']['unset'] == 1


class FakeDetector:
    patch_size = 14
    metadata = {}
    def __init__(self, name, records):
        self.name, self.records = name, records
    def fit(self, batches):
        self.records[self.name] = [batch.clone() for batch in batches]
    def save(self, path):
        Path(path).write_text(self.name)
    def predict(self, batch):
        return {'pred_score':torch.tensor([.5]), 'anomaly_map':torch.ones((1,1,*batch.shape[-2:]))*.5}


def test_engine_uses_identical_images_and_separate_artifacts(tmp_path):
    (tmp_path/'images').mkdir()
    rgb = np.random.default_rng(3).integers(0,256,(12,29,3),dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path/'images/image.png')
    item = dict(id='image',name='image.png',file='image.png',url='/original')
    task = dict(id='task',algorithm='comparison',image_size=None,alignment='pad',rotation=True,normal=[item],test=[item])
    records, progress = {}, []
    engine = DetectorEngine(dict(anomalydino=tmp_path,subspacead=tmp_path),'cpu')
    with patch('adkit.create_detector', side_effect=lambda name, **kw: FakeDetector(name, records)), patch('adkit.load_detector', side_effect=lambda name, *a, **kw: FakeDetector(name,records)):
        assert engine.execute(task,tmp_path,'fit',lambda **kw:progress.append(kw)) == {'model_ready':True}
        assert (tmp_path/'model_anomalydino.pt').is_file() and (tmp_path/'model_subspacead.pt').is_file()
        for first, second in zip(records['anomalydino'],records['subspacead']):
            torch.testing.assert_close(first, second)
        assert len(records['anomalydino']) == len(records['subspacead']) == 8
        engine.execute(task,tmp_path,'predict',lambda **kw:progress.append(kw))
    results = [p['result'] for p in progress if 'result' in p]
    assert len(results) == 2 and results[0]['raw_map'] != results[1]['raw_map']
    assert [p['completed'] for p in progress if 'completed' in p] == [1,2]
    for result in results:
        assert np.load(tmp_path/'results'/result['raw_map']).shape == (12,29)
        assert result['input_shape'] == [14,42]


def png(shape=(12,60)):
    file = BytesIO(); Image.new('RGB',(shape[1],shape[0]),'white').save(file,format='PNG')
    return file.getvalue()


def test_api_labels_thresholds_jobs_and_invalidation(tmp_path,monkeypatch):
    monkeypatch.setenv('ADKIT_DATA_DIR',str(tmp_path))
    records = {}
    with patch('adkit.create_detector',side_effect=lambda name, **kw:FakeDetector(name,records)), patch('adkit.load_detector',side_effect=lambda name,*a,**kw:FakeDetector(name,records)), TestClient(create_app()) as client:
        task = client.post('/api/tasks',json={'name':'compare','algorithm':'comparison','image_size':None}).json()
        base = f"/api/tasks/{task['id']}"
        def get(): return client.get(base).json()
        def job(operation):
            response = client.post(base+'/jobs/'+operation)
            assert response.status_code == 202
            for _ in range(100):
                if get()['job']['state'] not in ['queued','running']: break
                time.sleep(.02)
            assert get()['job']['state'] == 'done', get()['job']
        for kind in ['normal','test']:
            response = client.post(base+'/images/'+kind,files=[('files',('wide.png',png(),'image/png'))])
            assert response.status_code == 201
        id = get()['test'][0]['id']
        assert client.patch(base+f'/images/test/{id}/label',json={'label':'defect'}).status_code == 200
        job('fit'); job('predict')
        assert len(get()['results']) == 2 and get()['job']['total'] == 2
        for model, score in [('anomalydino',.4),('subspacead',.6)]:
            body = dict(algorithm=model,threshold=score,area_threshold=2)
            assert client.patch(base+'/threshold',json=body).status_code == 200
            report = client.post(base+'/assessment',json=body).json()
            assert report['counts']['detected' if score == .4 else 'false_negative'] == 1
        assert get()['thresholds']['anomalydino']['threshold'] == .4
        assert get()['thresholds']['subspacead']['threshold'] == .6
        assert client.patch(base+'/threshold',json={'threshold':.5}).status_code == 400
        assert client.get(get()['results'][0]['heatmap']).status_code == 200
        assert client.get(base+'/files/results/'+get()['results'][0]['raw_map']).status_code == 404
        # Labels change statistics without invalidating weights or predictions.
        client.patch(base+f'/images/test/{id}/label',json={'label':'normal'})
        report = client.post(base+'/assessment',json=dict(algorithm='anomalydino',threshold=.4,area_threshold=2)).json()
        assert report['counts']['false_positive'] == 1 and get()['model_ready']
        # Snapshot reuse keeps IDs/labels but owns separate files and no stale models.
        clone_response = client.post(base+'/comparison')
        assert clone_response.status_code == 201
        clone = clone_response.json()
        assert clone['algorithm'] == 'comparison' and not clone['model_ready']
        assert clone['test'][0]['id'] == id and clone['test'][0]['label'] == 'normal'
        assert clone['test'][0]['url'] != get()['test'][0]['url']
        assert client.get(clone['test'][0]['url']).status_code == 200
        client.post(base+'/images/normal',files=[('files',('second.png',png(),'image/png'))])
        assert not get()['model_ready'] and not get()['results']
        assert not list((tmp_path/task['id']).glob('model*.pt'))
