"""SuperADD numerical, geometry, offline lifecycle, and CLI regression tests."""
import ast
import itertools
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import timm
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from adkit import create_detector, load_detector
from adkit.superadd import SuperADDDetector, axis_patch_split, distance_coreset, euclidean_knn, gaussian_blur


@pytest.mark.parametrize('k', [1, 5, 9])
def test_euclidean_knn_matches_independent_dense(k):
    torch.manual_seed(3)
    q, bank = torch.randn(13, 7), torch.randn(19, 7)
    expected = torch.cdist(q, bank).topk(k, largest=False).values
    torch.testing.assert_close(euclidean_knn(q, bank, k, 4, 6), expected)


@pytest.mark.parametrize('size', [48, 64, 80, 112, 320])
def test_tile_ownership_matches_pinned_upstream(size):
    source = Path('references/super_add/torch_model.py').read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PatchedExecution')
    namespace = dict(nn=nn, torch=torch, itertools=itertools, math=math)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'upstream', 'exec'), namespace)
    reference = namespace['PatchedExecution'](nn.Identity(), 48, 16, 16)
    inputs, predictions, outputs = reference.axis_patch_split(size)
    actual = axis_patch_split(size, 48, 16, 16)
    assert actual == list(zip(inputs, predictions, outputs))
    covered = [i for _, _, (lo, hi) in actual for i in range(lo, hi)]
    assert covered == list(range(size // 16))


@pytest.mark.parametrize('kind', ['random', 'identical', 'duplicates'])
@pytest.mark.parametrize('iterations', [1, 4, 100])
def test_coreset_terminates_and_obeys_cap(kind, iterations):
    torch.manual_seed(7)
    x = torch.randn(33, 5)
    if kind == 'identical':
        x.zero_()
    elif kind == 'duplicates':
        x = x[:3].repeat(11, 1)
    selected = distance_coreset(x, 7, iterations, query_chunk=8, bank_chunk=9)
    assert selected.shape == (7, 5)
    assert torch.isfinite(selected).all()
    assert all((row == x).all(1).any() for row in selected)
    torch.testing.assert_close(distance_coreset(x, 40), x)


class GridEncoder(nn.Module):
    """Deterministic local token extractor to check tile stitching end to end."""
    num_features = 3
    def forward_intermediates(self, batch, **kwargs):
        tokens = F.avg_pool2d(batch, 16).flatten(2).transpose(1, 2)
        return [tokens, tokens * 2]


def test_stitching_keeps_non_square_pixels_and_batch_order():
    model = object.__new__(SuperADDDetector)
    model.encoder, model.layers = GridEncoder(), (0, 1)
    model.patch_size, model.tile_size, model.patch_overlap = 16, 48, 16
    model.device = torch.device('cpu')
    images = torch.arange(2 * 3 * 80 * 112).reshape(2, 3, 80, 112).float()
    expected = F.avg_pool2d(images, 16).permute(0, 2, 3, 1)
    actual = model.extract_features(images)
    torch.testing.assert_close(actual[0], expected)
    torch.testing.assert_close(actual[1], expected * 2)
    small = model.extract_features(images[:, :, :16, :32])
    assert small.shape == (2, 2, 1, 2, 3)


def test_map_and_score_match_dense_formula():
    model = object.__new__(SuperADDDetector)
    model.encoder, model.layers = GridEncoder(), (0, 1)
    model.patch_size, model.tile_size, model.patch_overlap = 16, 48, 16
    model.device = torch.device('cpu')
    model.query_chunk, model.bank_chunk = 3, 5
    model.gaussian_blur_sigma, model.score_quantile = 0, .1
    torch.manual_seed(12)
    model.memory_bank = torch.randn(2, 9, 3)
    images = torch.randn(2, 3, 80, 112)
    features = model.extract_features(images)
    maps = [torch.cdist(x.reshape(-1, 3), bank).min(1).values.reshape(2, 5, 7) / 3
            for x, bank in zip(features, model.memory_bank)]
    expected = F.interpolate(torch.stack(maps, 1), (80, 112), mode='bilinear', align_corners=False).mean(1, keepdim=True)
    actual = model.predict(images)
    torch.testing.assert_close(actual['anomaly_map'], expected)
    score = expected.flatten(1).topk(896, dim=1).values.mean(1)
    torch.testing.assert_close(actual['pred_score'], score)


def test_gaussian_matches_independent_2d_convolution():
    image = torch.rand(2, 1, 40, 50)
    sigma = 1.5
    axis = torch.arange(-6, 7).float()
    kernel = torch.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    expected = F.conv2d(F.pad(image, (6, 6, 6, 6), mode='reflect'), kernel[None, None])
    torch.testing.assert_close(gaussian_blur(image, sigma), expected)


@pytest.fixture
def tiny_model(tmp_path):
    torch.set_num_threads(2)
    original_create = timm.create_model
    def tiny(name, **kwargs):
        return original_create(name, embed_dim=32, depth=2, num_heads=4, **kwargs)
    weights = tmp_path / 'weights.pt'
    torch.save(tiny('vit_small_patch16_dinov3', pretrained=False, num_classes=0).state_dict(), weights)
    with patch('adkit.superadd.timm.create_model', side_effect=tiny), patch(
            'socket.create_connection', side_effect=AssertionError('Unexpected network')):
        yield create_detector('superadd', weights=weights, device='cpu', layers=[0, 1],
                              tile_size=48, patch_overlap=16, max_database_size=7,
                              subsampling_iterations=4, query_chunk=3, bank_chunk=4)


def test_tiny_real_dinov3_offline_roundtrip_and_refit(tiny_model, tmp_path):
    model = tiny_model
    images = torch.randn(2, 3, 63, 79)
    with pytest.raises(RuntimeError, match='empty'):
        model.predict(images)
    model.fit([images])
    assert model.memory_bank.shape == (2, 7, 32)
    assert not model.encoder.training
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    result = model.predict(images)
    assert result['anomaly_map'].shape == (2, 1, 63, 79)
    assert result['pred_score'].shape == (2,)
    assert torch.isfinite(result['anomaly_map']).all()
    model.metadata = {'data': {'image_size': None}}
    checkpoint = tmp_path / 'model.pt'
    model.save(checkpoint)
    loaded = load_detector('superadd', checkpoint, weights=model.weights)
    torch.testing.assert_close(result['anomaly_map'], loaded.predict(images)['anomaly_map'], rtol=0, atol=0)
    assert loaded.metadata == model.metadata
    loaded.fit([images[:1, :, :16, :16]])
    assert loaded.reference_patches == 1
    with pytest.raises(ValueError, match='No normal'):
        loaded.fit([])
    state = torch.load(checkpoint, weights_only=True)
    assert 'encoder' not in state
    state['memory_bank'] = torch.zeros(1, 7, 32)
    torch.save(state, tmp_path / 'bad.pt')
    with pytest.raises(ValueError, match='Invalid reference'):
        load_detector('superadd', tmp_path / 'bad.pt')
    state['weight_sha256'] = 'mismatch'
    torch.save(state, tmp_path / 'bad.pt')
    with pytest.raises(ValueError, match='checksum'):
        load_detector('superadd', tmp_path / 'bad.pt')


@pytest.mark.parametrize('params', [dict(score_quantile=0), dict(score_quantile=float('nan')),
                                    dict(subsampling_iterations=0), dict(gaussian_blur_sigma=float('inf')),
                                    dict(layers=[]), dict(layers=[1, 0]), dict(layers=[0, 0])])
def test_invalid_options_fail_before_loading_weights(params):
    with pytest.raises(ValueError):
        SuperADDDetector(weights='absent.pt', **params)


def test_cli_fit_uses_selected_model_stride(tmp_path):
    from adkit.run import run
    class Recorder:
        patch_size = 16
        device = 'cpu'
        weight_sha256 = 'test'
        reference_patches = 1
        def fit(self, batches):
            shapes.extend(tuple(batch.shape) for batch in batches)
        def save(self, path):
            Path(path).write_text('test')
    shapes = []
    normal = tmp_path / 'normal'
    normal.mkdir()
    Image.fromarray(np.zeros((17, 33, 3), dtype=np.uint8)).save(normal / 'sample.png')
    config = dict(model=dict(name='superadd'), data=dict(format='folder', normal_dir=str(normal), image_size=None),
                  run=dict(mode='fit'), output=dict(directory=str(tmp_path / 'output')))
    with patch('adkit.run.create_detector', return_value=Recorder()):
        run(config)
    assert shapes == [(1, 3, 32, 48)]
