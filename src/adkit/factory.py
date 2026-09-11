"""按算法名分发构造与加载操作，延迟导入可选算法依赖。"""
from pathlib import Path

from .base import BaseDetector


def _detector_class(name: str) -> type[BaseDetector]:
    """解析算法名，仅导入所选实现及其依赖。"""
    if name == "anomalydino":
        from .anomalydino import AnomalyDinoDetector
        return AnomalyDinoDetector
    if name == "subspacead":
        from .subspacead import SubspaceADDetector
        return SubspaceADDetector
    if name == "superadd":
        from .superadd import SuperADDDetector
        return SuperADDDetector
    raise ValueError(f"Unknown algorithm: {name!r}. Available: anomalydino, subspacead, superadd")


def create_detector(name: str, **kwargs) -> BaseDetector:
    """将关键字参数直接交给检测器构造函数，未知参数由构造函数拒绝。"""
    return _detector_class(name)(**kwargs)


def load_detector(name: str, path: str | Path, *, device: str = "cpu",
                  weights: str | Path | None = None) -> BaseDetector:
    """从检查点恢复算法，仅允许覆盖运行设备和本地权重路径。"""
    return _detector_class(name).load(path, device=device, weights=weights)
