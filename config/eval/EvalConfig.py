from dataclasses import dataclass
from typing import List, Optional
from omegaconf import MISSING
from omegaconf import II
from config.data.DatasetConfig import VisionModConfig, TextModConfig


@dataclass
class EvalConfig:
    num_samples_train: int = 20000
    max_iteration: int = 10000
    evaluate_lr: bool = True
    logging_frequency_lr: int = 50
    logging_frequency_plots: int = 50
    validation_every_n_epochs: int = 25


@dataclass
class EvalModuleConfig:
    _target_: str = MISSING
    vision_modality_config: Optional[VisionModConfig] = None
    text_modality_config: Optional[TextModConfig] = None
    max_iter: int = II("eval.max_iteration")
    evaluate_lr: bool = II("eval.evaluate_lr")
    logging_frequency_lr: int = II("eval.logging_frequency_lr")
    logging_frequency_plots: int = II("eval.logging_frequency_plots")


@dataclass
class OfflineEvalConfig:
    do_offline_eval: bool = MISSING
    ckpt_path: str = MISSING
    seed: int = 0


@dataclass
class OfflineEvalModuleConfig:
    _target_: str = MISSING
