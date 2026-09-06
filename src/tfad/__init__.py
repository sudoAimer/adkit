"""Small, offline-first anomaly detection library."""
from .base import BaseDetector


def create_detector(config, checkpoint=None):
    """Create/load an algorithm from the YAML's model section."""
    from .anomalydino import AnomalyDinoDetector
    config = dict(config)
    name = config.pop("name")
    algorithms = {'anomalydino': AnomalyDinoDetector}
    if name == 'subspacead':
        from .subspacead import SubspaceADDetector
        algorithms['subspacead'] = SubspaceADDetector
    if name not in algorithms:
        raise ValueError(f"Unknown algorithm: {name!r}. Available: anomalydino, subspacead")
    implementation = algorithms[name]
    if checkpoint is not None:
        return implementation.load(checkpoint, device=config.get('device', 'cpu'), weights=config.get('weights'))
    return implementation(**config)


__all__ = ["BaseDetector", "create_detector"]
