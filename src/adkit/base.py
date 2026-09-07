# 检测器公共生命周期接口。
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from torch import Tensor


class BaseDetector(ABC):
    @abstractmethod
    def fit(self, batches: Iterable[Tensor]) -> None:
        """使用正常图像批次重建参考状态，不执行反向传播。
        Replace reference state using normal image batches [B,3,H,W]."""

    @abstractmethod
    def predict(self, batch: Tensor) -> dict[str, Tensor]:
        """基于已建立的参考状态返回 CPU 异常分数和异常图。
        Return pred_score [B], anomaly_map [B,1,H,W]."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """原子保存构造参数、参考状态与权重校验信息，不保存骨干权重。
        Save fitted state, excluding separately supplied backbone weights."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, device: str = "cpu", **kwargs):
        """加载新格式检查点并校验参考状态，允许覆盖设备与权重路径。
        Restore a fitted detector."""
