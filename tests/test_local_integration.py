# 依赖本地权重的可选集成测试，不自动下载资源。
"""Optional integration test; uses existing local weights and never downloads."""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from adkit import create_detector
from adkit.anomalydino import AnomalyDinoDetector


WEIGHTS = Path(__file__).resolve().parents[1]/'weights/dinov2_vits14/model.safetensors'


@pytest.mark.skipif(not WEIGHTS.is_file(), reason='Local DINOv2 weights not prepared')
def test_offline_checkpoint_roundtrip_and_refit(tmp_path):
    """使用本地权重检查检查点往返、重复建库与哈希校验。"""
    torch.set_num_threads(4)
    with patch('socket.create_connection', side_effect=AssertionError('Unexpected network access')):
        detector = create_detector('anomalydino', weights=str(WEIGHTS), device='cpu')
        x = torch.randn(1, 3, 56, 56, generator=torch.Generator().manual_seed(2))
        with pytest.raises(RuntimeError, match='empty'):
            detector.predict(x)
        detector.fit([x])
        original = detector.predict(x)
        assert detector.memory_bank.shape == (16, 384)
        assert original['pred_score'].max() < 1e-5
        detector.fit([x])
        assert len(detector.memory_bank) == 16 # replace, never accidentally append
        checkpoint = tmp_path/'model.pt'
        detector.save(checkpoint)
        loaded = AnomalyDinoDetector.load(checkpoint, device='cpu')
        actual = loaded.predict(x)
        torch.testing.assert_close(actual['pred_score'], original['pred_score'], rtol=0, atol=0)
        torch.testing.assert_close(actual['anomaly_map'], original['anomaly_map'], rtol=0, atol=0)
        state = torch.load(checkpoint, weights_only=True)
        assert set(state) == {'format_version','algorithm','init_params','weight_sha256','memory_bank','metadata'}
        state['weight_sha256'] = 'invalid'
        torch.save(state, tmp_path/'invalid.pt')
        with pytest.raises(ValueError, match='checksum'):
            AnomalyDinoDetector.load(tmp_path/'invalid.pt', device='cpu')
