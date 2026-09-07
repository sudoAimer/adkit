"""Model identifiers are validated against the live registry by the service."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

PositiveSize = Annotated[int, Field(strict=True, gt=0)]
ModelID = Annotated[str, Field(pattern=r'^[a-z][a-z0-9_-]*$', max_length=100)]


class ModelSelection(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    algorithms: list[ModelID] = Field(min_length=1)

    @field_validator('algorithms')
    @classmethod
    def unique(cls, values):
        return list(dict.fromkeys(values))


class TaskCreate(ModelSelection):
    name: str = Field(min_length=1, max_length=60)
    image_size: PositiveSize | tuple[PositiveSize, PositiveSize] | None = 448
    rotation: bool = False


class ThresholdUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    algorithm: ModelID | None = None
    threshold: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    area_threshold: int = Field(default=0, ge=0, strict=True)


class LabelUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    label: Literal['normal', 'defect'] | None
