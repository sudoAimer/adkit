# SubspaceAD 的 PCA、采样和离线检查点回归测试。
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

from adkit import create_detector, load_detector
from adkit.data import prepare, select_reference, ReferenceBatches
from adkit.subspacead import fit_pca, reconstruction_scores, subspace_map


def test_pca_matches_dense_projection():
    """检查分批 PCA 与完整矩阵投影一致。"""
    x = np.random.default_rng(7).normal(size=(70,8)).astype(np.float32)
    state = fit_pca(lambda: iter(np.array_split(x,5)),components=4)
    mean = x.astype(np.float64).mean(0)
    _, _, v = np.linalg.svd(x.astype(np.float64)-mean,full_matrices=False)
    np.testing.assert_allclose(state['mu'],mean,atol=1e-12)
    np.testing.assert_allclose(state['components']@state['components'].T,v[:4].T@v[:4],atol=1e-12)
    expected = np.sum((x-(x-mean)@v[:4].T@v[:4]-mean)**2,axis=1)
    np.testing.assert_allclose(reconstruction_scores(x,state),expected,rtol=2e-6,atol=2e-6)


def test_pca_official_two_pass_equivalence():
    """检查两遍 PCA 与官方实现一致。"""
    path = Path(__file__).resolve().parents[1]/'references/SubspaceAD/src/subspacead/core/pca.py'
    if not path.exists():
        pytest.skip('Official reference source absent')
    spec = importlib.util.spec_from_file_location('reference_pca',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = np.random.default_rng(8).normal(size=(50,6)).astype(np.float32)
    def source():
        """每次调用重新遍历特征源，为两遍 PCA 统计提供数据。"""
        return iter(np.array_split(data,5))
    reference = module.PCAModel(ev=.9)
    reference.device = torch.device('cpu')
    expected = reference.fit(source,6,50,5)
    actual = fit_pca(source,explained_variance=.9)
    assert actual['k'] == expected['k']
    np.testing.assert_array_equal(actual['mu'],expected['mu'])
    np.testing.assert_array_equal(actual['components'],expected['components'])


def test_subspace_reference_selection():
    """检查 SubspaceAD 的种子打乱采样协议。"""
    paths = [f'{i:03}.png' for i in range(209)]
    rng = random.Random(42)
    rng.shuffle(paths)
    assert select_reference(sorted(paths),1,42,'subspacead') == paths[:1]


def test_invalid_pca_passes():
    """检查空数据、零方差及两遍样本数量不一致的错误处理。"""
    with pytest.raises(ValueError,match='at least two'):
        fit_pca(lambda: iter([]))
    with pytest.raises(ValueError,match='zero variance'):
        fit_pca(lambda: iter([np.ones((5,3))]))
    passes = iter([[np.ones((4,3))],[np.ones((3,3))]])
    with pytest.raises(ValueError,match='same number'):
        fit_pca(lambda: iter(next(passes)))


def test_tiny_offline_subspace_roundtrip(tmp_path):
    """使用微型离线骨干检查 SubspaceAD 保存加载与资产校验。"""
    torch.set_num_threads(2)
    asset = tmp_path/'weights'
    config = Dinov2WithRegistersConfig(hidden_size=16,num_hidden_layers=2,num_attention_heads=4,
                                      image_size=28,patch_size=14,num_register_tokens=2,mlp_ratio=2)
    Dinov2WithRegistersModel(config).save_pretrained(asset)
    BitImageProcessor().save_pretrained(asset)
    options = dict(name='subspacead',weights=str(asset),device='cpu',layers=[-1,-2],components=3)
    with patch('socket.create_connection',side_effect=AssertionError('Unexpected network')):
        model = create_detector(**options)
        image = np.random.default_rng(2).integers(0,256,size=(35,51,3),dtype=np.uint8)
        tensor = prepare(image,28,processor=model.processor,alignment='legacy').unsqueeze(0)
        assert tensor.shape == (1,3,28,28)
        with pytest.raises(RuntimeError,match='empty'):
            model.predict(tensor)
        # List is re-iterable; generator is deliberately also supported.
        model.fit(iter([tensor,tensor.flip(-1)]))
        assert model.reference_patches == 8
        result = model.predict(tensor)
        assert result['anomaly_map'].shape == (1,1,28,28)
        assert torch.isfinite(result['anomaly_map']).all()
        arbitrary = model.predict(torch.randn(1, 3, 13, 29))
        assert arbitrary['anomaly_map'].shape == (1, 1, 13, 29)
        assert torch.isfinite(arbitrary['anomaly_map']).all()
        model.save(tmp_path/'model.pt')
        saved = torch.load(tmp_path/'model.pt',weights_only=True)
        assert 'memory_bank' not in saved and 'encoder' not in saved
        loaded = load_detector('subspacead', tmp_path/'model.pt', device='cpu', weights=asset)
        torch.testing.assert_close(result['anomaly_map'],loaded.predict(tensor)['anomaly_map'],rtol=0,atol=0)
        loaded.fit([tensor])
        assert loaded.reference_patches == 4
        processor_path = asset/'preprocessor_config.json'
        processor_config = json.loads(processor_path.read_text())
        processor_config['rescale_factor'] = .5
        processor_path.write_text(json.dumps(processor_config))
        with pytest.raises(ValueError,match='processor config checksum'):
            load_detector('subspacead', tmp_path/'model.pt', device='cpu', weights=asset)


def test_two_pass_augmentations_regenerate(tmp_path):
    """检查两遍迭代保留原图并重新生成随机增强。"""
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
    """检查异常图输出尺寸符合目标尺寸。"""
    assert subspace_map(np.ones((3,4)),(42,56)).shape == (42,56)


def test_fullshot_has_no_fewshot_augmentation(tmp_path):
    """检查全样本模式不执行少样本增强。"""
    image = tmp_path/'normal.png'
    Image.fromarray(np.zeros((28,28,3),dtype=np.uint8)).save(image)
    source = ReferenceBatches([image],{'image_size':28,'shots':-1,'augmentation':'subspacead','aug_count':30})
    assert len(list(source)) == 1


@pytest.fixture
def shared_small_weights(tmp_path):
    """Actual timm DINOv2 blocks, reduced width but 12-layer Small indexing."""
    import timm
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    original = timm.create_model
    def tiny(name, **kwargs):
        return original(name, embed_dim=32, depth=12, num_heads=4, **kwargs)
    weights = tmp_path/'shared.safetensors'
    save_file(tiny('vit_small_patch14_dinov2.lvd142m', pretrained=False,
                   num_classes=0, dynamic_img_size=True).state_dict(), str(weights))
    with patch('timm.create_model', side_effect=tiny), patch(
            'socket.create_connection', side_effect=AssertionError('Unexpected network')):
        yield weights


@pytest.mark.parametrize('positional_encoding', ['official', 'timm'])
def test_small_shares_anomalydino_weights_and_preserves_intermediate_features(shared_small_weights, positional_encoding):
    from adkit.dinov2 import input_tokens
    options = dict(weights=shared_small_weights, device='cpu', positional_encoding=positional_encoding)
    subspace = create_detector('subspacead', **options)
    anomaly = create_detector('anomalydino', **options)
    assert subspace.backend == 'timm'
    assert subspace.layers == (-4, -5)
    assert subspace.processor is None
    assert subspace.patch_size == anomaly.patch_size == 14
    assert subspace.weight_sha256 == anomaly.weight_sha256
    assert subspace.feature_dim == anomaly.encoder.num_features
    for name, value in anomaly.encoder.state_dict().items():
        torch.testing.assert_close(value, subspace.encoder.state_dict()[name], rtol=0, atol=0)
    captured = {}
    handles = [anomaly.encoder.blocks[i].register_forward_hook(
        lambda module, args, result, i=i: captured.__setitem__(i, result.detach().clone())) for i in [7, 8, 11]]
    batch = torch.randn(2, 3, 28, 42)
    try:
        final = anomaly.extract_features(batch)
    finally:
        for handle in handles:
            handle.remove()
    # 13 HF hidden states: -4 is state 9/block index 8, -5 is state 8/index 7.
    expected = torch.stack([captured[8][:, 1:], captured[7][:, 1:]]).mean(0)
    torch.testing.assert_close(subspace.extract_features(batch), expected, rtol=0, atol=0)
    torch.testing.assert_close(final, anomaly.encoder.norm(captured[11])[:, 1:], rtol=0, atol=0)
    assert not subspace.encoder.training
    assert not any(p.requires_grad for p in subspace.encoder.parameters())


def test_small_checkpoint_refit_and_wrong_weights(shared_small_weights, tmp_path):
    options = dict(weights=shared_small_weights, device='cpu', components=3)
    model = create_detector('subspacead', **options)
    batch = torch.randn(1, 3, 27, 41)
    model.fit([batch, batch.flip(-1)])
    assert model.reference_patches == 12
    result = model.predict(batch)
    assert result['anomaly_map'].shape == (1, 1, 27, 41)
    assert torch.isfinite(result['anomaly_map']).all()
    checkpoint = tmp_path/'small.pt'
    model.save(checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved['init_params']['backend'] == 'timm'
    assert saved['init_params']['encoder_name'] == 'vit_small_patch14_dinov2.lvd142m'
    assert saved['asset_hashes'] == {}
    restored = load_detector('subspacead', checkpoint, weights=shared_small_weights)
    torch.testing.assert_close(result['anomaly_map'], restored.predict(batch)['anomaly_map'], rtol=0, atol=0)
    restored.fit(iter([batch]))
    assert restored.reference_patches == 6
    saved['weight_sha256'] = 'wrong'
    torch.save(saved, tmp_path/'wrong.pt')
    with pytest.raises(ValueError, match='checksum'):
        load_detector('subspacead', tmp_path/'wrong.pt')
    with pytest.raises(ValueError, match='hidden states'):
        create_detector('subspacead', **options, layers=[-18])


def test_legacy_transformers_checkpoint_still_loads(tmp_path):
    asset = tmp_path/'legacy'
    config = Dinov2WithRegistersConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=4,
                                      image_size=28, patch_size=14, num_register_tokens=2, mlp_ratio=2)
    Dinov2WithRegistersModel(config).save_pretrained(asset)
    BitImageProcessor().save_pretrained(asset)
    model = create_detector('subspacead', weights=asset, device='cpu', layers=[-1, -2], components=3)
    batch = torch.randn(1, 3, 28, 28)
    model.fit([batch])
    checkpoint = tmp_path/'legacy.pt'
    model.save(checkpoint)
    state = torch.load(checkpoint, weights_only=True)
    for name in ['backend', 'encoder_name', 'positional_encoding']:
        state['init_params'].pop(name)
    torch.save(state, checkpoint)
    restored = load_detector('subspacead', checkpoint)
    assert restored.backend == 'transformers'
    torch.testing.assert_close(model.predict(batch)['anomaly_map'], restored.predict(batch)['anomaly_map'], rtol=0, atol=0)


def test_web_registry_shares_default_small_asset():
    from backend.registry import MODEL_REGISTRY
    assert MODEL_REGISTRY['subspacead'].default_weights == MODEL_REGISTRY['anomalydino'].default_weights
    assert MODEL_REGISTRY['subspacead'].assets == ()
