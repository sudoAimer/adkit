"""Server-owned model registry. UI and requests contain IDs, never weight paths."""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    weights_env: str
    default_weights: str
    assets: tuple[str, ...] = ()
    parameters: dict = field(default_factory=dict)
    create: Callable | None = None
    load: Callable | None = None

    def __post_init__(self):
        if not re.fullmatch(r'[a-z][a-z0-9_-]*', self.id):
            raise ValueError('Model ID must be a safe lowercase identifier')

    def weights(self, root):
        return Path(os.getenv(self.weights_env, str(root / self.default_weights)))

    def available(self, path):
        return all((path / asset).is_file() for asset in self.assets) if self.assets else path.is_file()

    def build(self, weights, device, parameters=None):
        from adkit import create_detector
        parameters = dict(self.parameters if parameters is None else parameters)
        return self.create(weights=weights, device=device, **parameters) if self.create else create_detector(
            self.id, weights=weights, device=device, **parameters)

    def restore(self, checkpoint, weights, device):
        from adkit import load_detector
        return self.load(checkpoint, weights=weights, device=device) if self.load else load_detector(
            self.id, checkpoint, weights=weights, device=device)


MODEL_REGISTRY = {
    spec.id: spec for spec in [
        ModelSpec('anomalydino', 'AnomalyDINO', 'ADKIT_ANOMALYDINO_WEIGHTS',
                  'weights/dinov2_vits14/model.safetensors'),
        ModelSpec('subspacead', 'SubspaceAD', 'ADKIT_SUBSPACEAD_WEIGHTS',
                  'weights/dinov2_with_registers_giant',
                  ('model.safetensors', 'config.json', 'preprocessor_config.json')),
    ]
}
