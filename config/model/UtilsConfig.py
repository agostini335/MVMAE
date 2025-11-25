from dataclasses import dataclass
from omegaconf import MISSING
from omegaconf import II
from typing import Optional


@dataclass
class AnnealingConfig:
    name: Optional[str] = None
    init_beta: float = 0.0
    final_beta: float = 1.0
    start_annealing: int = II("training.n_epochs_warmup")
    end_annealing: int = II("training.n_epochs")


@dataclass
class AnnealingModuleConfig:
    _target_: Optional[str] = MISSING
    init_beta: float = II("annealing.init_beta")
    final_beta: float = II("annealing.final_beta")
    start_annealing: int = II("annealing.start_annealing")
    end_annealing: int = II("annealing.end_annealing")


@dataclass
class SigmoidAnnealingConfig(AnnealingConfig):
    name: str = "sigmoid"
    steepness: float = 0.02


@dataclass
class SigmoidAnnealingModuleConfig(AnnealingModuleConfig):
    _target_: str = "mm_maes.utils.SigmoidAnnealing"
    # name: str = "sigmoid"
    steepness: float = 0.02
    # steepness: float = II("annealing.steepness")


@dataclass
class ExpAnnealingConfig(AnnealingConfig):
    name: str = "exp"


@dataclass
class ExpAnnealingModuleConfig(AnnealingModuleConfig):
    _target_: str = "mm_maes.utils.ExpAnnealing"


@dataclass
class CosAnnealingConfig(AnnealingConfig):
    name: str = "cos"


@dataclass
class CosAnnealingModuleConfig(AnnealingModuleConfig):
    _target_: str = "mm_maes.utils.CosAnnealing"


@dataclass
class LinearAnnealingConfig(AnnealingConfig):
    name: str = "linear"


@dataclass
class LinearAnnealingModuleConfig(AnnealingModuleConfig):
    _target_: str = "mm_maes.utils.LinearAnnealing"


@dataclass
class NoAnnealingConfig(AnnealingConfig):
    name: str = "noannealing"


@dataclass
class NoAnnealingModuleConfig(AnnealingModuleConfig):
    _target_: str = "mm_maes.utils.NoAnnealing"
