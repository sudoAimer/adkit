import importlib.util
import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import BitImageProcessor, Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from tfad import create_detector
from tfad.data import prepare, select_reference, ReferenceBatches
from tfad.subspacead import fit_pca, reconstruction_scores, subspace_map


def test_pca_matches_dense_projection():
    x = np.random.default_rng(7).normal(size=(70,8)).astype(np.float32)
    state = fit_pca(lambda: iter(np.array_split(x,5)),components=4)
    mean = x.astype(np.float64).mean(0)
    _, _, v = np.linalg.svd(x.astype(np.float64)-mean,full_matrices=False)
    np.testing.assert_allclose(state['mu'],mean,atol=1e-12)
    np.testing.assert_allclose(state['components']@state['components'].T,v[:4].T@v[:4],atol=1e-12)
    expected = np.sum((x-(x-mean)@v[:4].T@v[:4]-mean)**2,axis=1)
    np.testing.assert_allclose(reconstruction_scores(x,state),expected,rtol=2e-6,atol=2e-6)


def test_pca_official_two_pass_equivalence():
    path = Path(__file__).resolve().parents[1]/'references/SubspaceAD/src/subspacead/core/pca.py'
    if not path.exists():
        pytest.skip('Official reference source absent')
    spec = importlib.util.spec_from_file_location('reference_pca',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = np.random.default_rng(8).normal(size=(50,6)).astype(np.float32)
    def source():
        return iter(np.array_split(data,5))
    reference = module.PCAModel(ev=.9)
    reference.device = torch.device('cpu')
    expected = reference.fit(source,6,50,5)
    actual = fit_pca(source,explained_variance=.9)
    assert actual['k'] == expected['k']
    np.testing.assert_array_equal(actual['mu'],expected['mu'])
    np.testing.assert_array_equal(actual['components'],expected['components'])


def test_subspace_reference_selection():
    paths = [f'{i:03}.png' for i in range(209)]
    rng = random.Random(42)
    rng.shuffle(paths)
    assert select_reference(sorted(paths),1,42,'subspacead') == paths[:1]


def test_invalid_pca_passes():
    with pytest.raises(ValueError,match='at least two'):
        fit_pca(lambda: iter([]))
    with pytest.raises(ValueError,match='zero variance'):
        fit_pca(lambda: iter([np.ones((5,3))]))
    passes = iter([[np.ones((4,3))],[np.ones((3,3))]])
    with pytest.raises(ValueError,match='same number'):
        fit_pca(lambda: iter(next(passes)))


def test_tiny_offline_subspace_roundtrip(tmp_path):
    torch.set_num_threads(2)
    asset = tmp_path/'weights'
    config = Dinov2WithRegistersConfig(hidden_size=16,num_hidden_layers=2,num_attention_heads=4,
                                      image_size=28,patch_size=14,num_register_tokens=2,mlp_ratio=2)
    Dinov2WithRegistersModel(config).save_pretrained(asset)
    BitImageProcessor().save_pretrained(asset)
    options = dict(name='subspacead',weights=str(asset),device='cpu',layers=[-1,-2],components=3)
    with patch('socket.create_connection',side_effect=AssertionError('Unexpected network')):
        model = create_detector(options)
        image = np.random.default_rng(2).integers(0,256,size=(35,51,3),dtype=np.uint8)
        tensor = prepare(image,28,processor=model.processor).unsqueeze(0)
        assert tensor.shape == (1,3,28,28)
        with pytest.raises(RuntimeError,match='empty'):
            model.predict(tensor)
        # List is re-iterable; generator is deliberately also supported.
        model.fit(iter([tensor,tensor.flip(-1)]))
        assert model.reference_patches == 8
        result = model.predict(tensor)
        assert result['anomaly_map'].shape == (1,1,28,28)
        assert torch.isfinite(result['anomaly_map']).all()
        model.save(tmp_path/'model.pt')
        saved = torch.load(tmp_path/'model.pt',weights_only=True)
        assert 'memory_bank' not in saved and 'encoder' not in saved
        loaded = create_detector(options,checkpoint=tmp_path/'model.pt')
        torch.testing.assert_close(result['anomaly_map'],loaded.predict(tensor)['anomaly_map'],rtol=0,atol=0)
        loaded.fit([tensor])
        assert loaded.reference_patches == 4
        processor_path = asset/'preprocessor_config.json'
        processor_config = json.loads(processor_path.read_text())
        processor_config['rescale_factor'] = .5
        processor_path.write_text(json.dumps(processor_config))
        with pytest.raises(ValueError,match='processor config checksum'):
            create_detector(options,checkpoint=tmp_path/'model.pt')


def test_two_pass_augmentations_regenerate(tmp_path):
    file = tmp_path/'image.png'
    image = np.random.default_rng(4).integers(0,256,size=(56,56,3),dtype=np.uint8)
    Image.fromarray(image).save(file)
    source = ReferenceBatches([file],{'image_size':56,'augmentation':'subspacead','aug_count':2})
    torch.manual_seed(42)
    first, second = list(source), list(source)
    assert len(first) == len(second) == 3
    assert torch.equal(first[0],second[0])
    assert not torch.equal(first[1],second[1])


def test_map_preserves_shape():
    assert subspace_map(np.ones((3,4)),(42,56)).shape == (42,56)


def test_fullshot_has_no_fewshot_augmentation(tmp_path):
    image = tmp_path/'normal.png'
    Image.fromarray(np.zeros((28,28,3),dtype=np.uint8)).save(image)
    source = ReferenceBatches([image],{'image_size':28,'shots':-1,'augmentation':'subspacead','aug_count':30})
    assert len(list(source)) == 1
