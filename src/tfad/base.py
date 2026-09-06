from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from torch import Tensor


class BaseDetector(ABC):
    @abstractmethod
    def fit(self, batches: Iterable[Tensor]) -> None:
        """Replace reference state using normal image batches [B,3,H,W]."""

    @abstractmethod
    def predict(self, batch: Tensor) -> dict[str, Tensor]:
        """Return pred_score [B], anomaly_map [B,1,H,W]."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save fitted state, excluding separately supplied backbone weights."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, device: str = "cpu", **kwargs):
        """Restore a fitted detector."""
