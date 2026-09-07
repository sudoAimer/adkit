"""请求参数校验；网页不能传入任意骨干路径或算法参数。"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskCreate(BaseModel):
    """固定任务的算法与预处理参数，避免建库和预测配置不一致。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=60)
    algorithm: Literal["anomalydino", "subspacead"] = "anomalydino"
    image_size: int = Field(default=448, ge=112, le=1008)
    rotation: bool = False

    @field_validator("image_size")
    @classmethod
    def check_patch_size(cls, value: int) -> int:
        """两个内置骨干均使用 14 像素 patch，提前拒绝不整除尺寸。"""
        if value % 14:
            raise ValueError("图像尺寸必须是 14 的整数倍")
        return value


class ThresholdUpdate(BaseModel):
    """阈值作用于原始图像分数；空值表示尚未设置判定阈值。"""

    model_config = ConfigDict(extra="forbid")
    threshold: float | None = Field(default=None, ge=0, allow_inf_nan=False)
