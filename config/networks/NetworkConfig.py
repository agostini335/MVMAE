from dataclasses import dataclass
from omegaconf import II
from omegaconf import MISSING


@dataclass
class EncoderModuleConfig:
    _target_: str = MISSING

@dataclass
class TextEncoderModuleConfig(EncoderModuleConfig):
    _target_: str = MISSING
    emb_dim: int = MISSING
    random_init: bool = False
    hidden_state_index: int = -1 # -1 means last hidden state
    freeze_all: bool = False


@dataclass
class DecoderModuleConfig:
    _target_: str = MISSING


@dataclass
class TokenizerModuleConfig:
    pretrained_model_name: str = MISSING
    mlm_probability: float = MISSING
    max_len: int = MISSING
    truncation: bool = MISSING


@dataclass
class MMNetworksModuleConfig:
    _target_: str = MISSING
    num_views: int = II(
        "dataset.num_views"
    )
    encoder_cfg: EncoderModuleConfig = MISSING
    decoder_cfg: DecoderModuleConfig = MISSING


@dataclass
class MMNetworksSharedWeightsModuleConfig(MMNetworksModuleConfig):
    _target_: str = "utils.vae.get_networks_shared_weights"


@dataclass
class MMNetworksIndependentWeightsModuleConfig(MMNetworksModuleConfig):
    _target_: str = "utils.vae.get_networks_shared_weights"
