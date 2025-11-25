from dataclasses import dataclass
from omegaconf import MISSING
from omegaconf import II
from typing import List, Dict, Optional
from config.data.DatasetConfig import VisionModConfig
from config.data.DatasetConfig import TextModConfig


@dataclass
class ModelConfig:
    offline_eval: bool = False
    # if possible: weight sharing between encoder and decoder of the different modalities (vision only)
    share_weights: bool = True
    share_weights_text: bool = True

@dataclass
class ModelModuleConfig:
    _target_: str = MISSING
    # modality_names: List[str] = II("dataset.modality_names")
    num_samples_lr_train: int = II("eval.num_samples_train")
    learning_rate: float = II("training.base_lr")
    vision_modality_config: Optional[VisionModConfig] = None
    text_modality_config: Optional[TextModConfig] = None

@dataclass
class ModelModuleConfig_MAE(ModelModuleConfig):
    encoders = MISSING
    decoders = MISSING
    evaluator = MISSING
    offline_eval: bool = II("model.offline_eval")

