"""Padding must preserve tiny/rectangular inputs and remove only added pixels."""
import numpy as np
import pytest
import torch
from transformers import BitImageProcessor
from adkit.data import prepare, restore_map, ReferenceBatches
from PIL import Image


@pytest.mark.parametrize('shape,expected', [((12,12),(14,14)), ((1,1),(14,14)),
    ((225,319),(238,322)), ((28,42),(28,42)), ((3,101),(14,112))])
def test_native_padding(shape, expected):
    rgb = np.random.default_rng(4).integers(0, 256, (*shape, 3), dtype=np.uint8)
    tensor, geometry = prepare(rgb, None, return_geometry=True)
    assert tuple(tensor.shape) == (3, *expected)
    assert prepare(rgb, None, processor=BitImageProcessor()).shape == tensor.shape
    unnormalized = tensor[:, :shape[0], :shape[1]] * torch.tensor([.229,.224,.225])[:,None,None] + torch.tensor([.485,.456,.406])[:,None,None]
    np.testing.assert_allclose(unnormalized.permute(1,2,0), rgb/255, atol=1e-7)
    amap = np.ones(expected, dtype=np.float32)*99
    amap[:shape[0], :shape[1]] = np.arange(shape[0]*shape[1]).reshape(shape)
    np.testing.assert_array_equal(restore_map(amap, geometry), amap[:shape[0], :shape[1]])


def test_shared_processor_geometry_and_explicit_size():
    rgb = np.zeros((101,157,3), dtype=np.uint8)
    for size, expected in [(56, (3,56,98)), ((12,29), (3,14,42)), (None, (3,112,168))]:
        plain = prepare(rgb, size)
        processed = prepare(rgb, size, processor=BitImageProcessor())
        assert plain.shape == processed.shape == expected


@pytest.mark.parametrize('size', [0, -1, 1.5, [1,0], [12], True])
def test_invalid_size(size):
    with pytest.raises(ValueError):
        prepare(np.zeros((12,12,3), dtype=np.uint8), size)


def test_mixed_sizes_do_not_stack_incompatible_tensors(tmp_path):
    paths = []
    for i, shape in enumerate([(12,12), (13,13), (20,40)]):
        path = tmp_path/f'{i}.png'
        Image.fromarray(np.zeros((*shape,3), dtype=np.uint8)).save(path)
        paths.append(path)
    batches = list(ReferenceBatches(paths, {'image_size': None}, batch_size=4))
    assert [tuple(x.shape) for x in batches] == [(2,3,14,14), (1,3,28,42)]


@pytest.mark.parametrize('encoding', ['official', 'timm'])
def test_tiny_anomalydino_arbitrary_tensor(tmp_path, encoding):
    from unittest.mock import patch
    from timm.models.vision_transformer import VisionTransformer
    from adkit.anomalydino import AnomalyDinoDetector
    torch.set_num_threads(2)
    def backbone(*args, **kwargs):
        return VisionTransformer(img_size=28,patch_size=14,embed_dim=16,depth=1,
                                 num_heads=2,num_classes=0,dynamic_img_size=True)
    weights = tmp_path/'tiny.pt'
    torch.save(backbone().state_dict(), weights)
    with patch('adkit.anomalydino.timm.create_model', side_effect=backbone):
        model = AnomalyDinoDetector(weights,device='cpu',positional_encoding=encoding)
        batch = torch.randn(1,3,12,29)
        model.fit([batch])
        prediction = model.predict(batch)
        assert prediction['anomaly_map'].shape == (1,1,12,29)
        assert prediction['patch_map'].shape == (1,1,1,3)
        assert torch.isfinite(prediction['anomaly_map']).all()
        assert prediction['pred_score'][0] < 1e-5


def test_legacy_processor_matches_historical_call():
    rgb = np.random.default_rng(0).integers(0,256,(51,72,3),dtype=np.uint8)
    processor = BitImageProcessor()
    expected = processor(images=[Image.fromarray(rgb)],return_tensors='pt',do_resize=True,
        size={'height':28,'width':28},do_center_crop=False,crop_size={'height':28,'width':28}).pixel_values[0]
    actual = prepare(rgb,28,processor=processor,alignment='legacy')
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
