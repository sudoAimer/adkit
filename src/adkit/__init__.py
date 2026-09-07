"""adkit 公共接口：检测器基类、显式参数工厂和检查点加载入口。"""
from .base import BaseDetector
from .factory import create_detector, load_detector

__all__ = ["BaseDetector", "create_detector", "load_detector"]
