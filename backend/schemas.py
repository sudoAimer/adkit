"""Validated workbench requests; model paths remain server-owned."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field

Algorithm = Literal['anomalydino', 'subspacead']
PositiveSize = Annotated[int, Field(strict=True, gt=0)]


class TaskCreate(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=60)
    algorithm: Literal['anomalydino', 'subspacead', 'comparison'] = 'anomalydino'
    image_size: PositiveSize | tuple[PositiveSize, PositiveSize] | None = 448
    rotation: bool = False


class ThresholdUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    algorithm: Algorithm | None = None
    threshold: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    area_threshold: int = Field(default=0, ge=0, strict=True)


class LabelUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    label: Literal['normal', 'defect'] | None
