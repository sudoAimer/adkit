import numpy as np
import pytest
import torch
from torch.nn import functional as F

from tfad import create_detector
from tfad.anomalydino import exact_knn, render_map
from tfad.data import prepare, select_reference
from tfad.metrics import aupro, evaluate


@pytest.mark.parametrize('k', [1, 3, 7])
def test_chunked_knn_matches_dense(k):
    generator = torch.Generator().manual_seed(42)
    query = F.normalize(torch.randn(13, 8, generator=generator), dim=-1)
    bank = F.normalize(torch.randn(19, 8, generator=generator), dim=-1)
    expected = (1-query@bank.T).clamp(0, 2).topk(k, largest=False).values.mean(1)
    actual = exact_knn(query, bank, k, query_chunk=4, bank_chunk=5)
    torch.testing.assert_close(actual, expected)


def test_invalid_algorithm_and_empty_bank():
    with pytest.raises(ValueError, match='Unknown algorithm'):
        create_detector({'name': 'missing'})
    with pytest.raises(ValueError):
        exact_knn(torch.ones(1, 3), torch.empty(0, 3))


def test_official_reference_selection():
    paths = ['002.png', '000.png', '001.png']
    assert [select_reference(paths, 1, seed)[0] for seed in range(3)] == sorted(paths)
    assert len(select_reference(paths, -1, 0)) == 3
    with pytest.raises(ValueError):
        select_reference(paths, 2, 1)


def test_non_square_geometry():
    image = np.zeros((101, 157, 3), dtype=np.uint8)
    tensor = prepare(image, 56)
    assert tensor.shape == (3, 56, 84)
    patch = np.zeros((4, 6), dtype=np.float32)
    patch[1, 4] = 1
    result = render_map(patch, image.shape[:2], sigma=0)
    assert result.shape == (101, 157)
    y, x = np.unravel_index(result.argmax(), result.shape)
    assert 100 < x < 130 and 25 < y < 50


def test_metrics_perfect_and_tied():
    masks = [np.zeros((8, 8), dtype=bool), np.zeros((8, 8), dtype=bool)]
    masks[1][2:4, 2:4] = True
    maps = [m.astype(np.float32) for m in masks]
    result = evaluate([0, 1], [0., 1.], maps, masks)
    assert all(value == pytest.approx(1.) for value in result.values())
    tied = [np.zeros_like(m, dtype=np.float32) for m in masks]
    # A constant predictor traces PRO=FPR; integral/0.3 = 0.15.
    assert aupro(tied, masks, .3) == pytest.approx(.15)


def test_metrics_reject_invalid_input():
    with pytest.raises(ValueError, match='both normal'):
        evaluate([0], [0.], [np.zeros((2, 2))], [np.zeros((2, 2))])
    with pytest.raises(ValueError):
        aupro([], [], limit=0)
