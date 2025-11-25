from dataclasses import dataclass, field
from omegaconf import II
from omegaconf import MISSING
from omegaconf import OmegaConf

from config.networks.NetworkConfig import EncoderModuleConfig
from config.networks.NetworkConfig import DecoderModuleConfig

# Register a resolver for length
OmegaConf.register_new_resolver("len", lambda x: len(x))






@dataclass
class ViTtinyEncoderModuleConfig(EncoderModuleConfig):
    _target_: str = "networks.NetworksMAE.MAE_Encoder"
    patch_size: int = II("model.patch_size")
    emb_dim: int = 192
    n_enc_layers: int = 12
    n_enc_heads: int = 3
    image_size: int = II("dataset.img_size")
    num_channels: int = II("dataset.image_channels")
    mask_ratio: float = II("model.mask_ratio")
    mlp_ratio: int = 4
    qk_norm: bool = True
    mod_embedding: int = "${len:${vision_modality.view_indices}}"
    log_debug: bool = II("log.debug")


@dataclass
class ViTtinyDecoderModuleConfig(DecoderModuleConfig):
    _target_: str = "networks.NetworksMAE.MAE_Decoder"
    patch_size: int = II("model.patch_size")
    emb_dim: int = 192
    n_dec_layers: int = 4
    n_dec_heads: int = 3
    image_size: int = II("dataset.img_size")
    num_channels: int = II("dataset.image_channels")
    mlp_ratio: int = 4
    qk_norm: bool = True
    mod_embedding: int = "${len:${vision_modality.view_indices}}"
    use_layer_norm: bool = True


@dataclass
class ViTbaseEncoderModuleConfig(EncoderModuleConfig):
    _target_: str = "networks.NetworksMAE.MAE_Encoder"
    patch_size: int = II("model.patch_size")
    emb_dim: int = 768
    n_enc_layers: int = 12
    n_enc_heads: int = 12
    image_size: int = II("dataset.img_size")
    num_channels: int = II("dataset.image_channels")
    mask_ratio: float = II("model.mask_ratio")
    mlp_ratio: int = 4
    qk_norm: bool = True
    mod_embedding: int = "${len:${vision_modality.view_indices}}"
    log_debug: bool = II("log.debug")


@dataclass
class ViTbaseDecoderModuleConfig(DecoderModuleConfig):
    _target_: str = "networks.NetworksMAE.MAE_Decoder"
    patch_size: int = II("model.patch_size")
    emb_dim: int = 768
    n_dec_layers: int = 4
    n_dec_heads: int = 12
    image_size: int = II("dataset.img_size")
    num_channels: int = II("dataset.image_channels")
    mlp_ratio: int = 4
    qk_norm: bool = True
    mod_embedding: int = "${len:${vision_modality.view_indices}}"
    use_layer_norm: bool = True


@dataclass
class ViTlargeEncoderModuleConfig(EncoderModuleConfig):
    _target_: str = "networks.NetworksMAE.MAE_Encoder"
    patch_size: int = II("model.patch_size")
    emb_dim: int = 1024
    n_enc_layers: int = 24
    n_enc_heads: int = 16
    image_size: int = II("dataset.img_size")
    num_channels: int = II("dataset.image_channels")
    mask_ratio: float = II("model.mask_ratio")
    mlp_ratio: int = 4
    qk_norm: bool = True
    mod_embedding: int = "${len:${vision_modality.view_indices}}"
    log_debug: bool = II("log.debug")


@dataclass
class ViTlargeDecoderModuleConfig(DecoderModuleConfig):
    _target_: str = "networks.NetworksMAE.MAE_Decoder"
    patch_size: int = II("model.patch_size")
    emb_dim: int = 1024
    n_dec_layers: int = 8
    n_dec_heads: int = 8
    image_size: int = II("dataset.img_size")
    num_channels: int = II("dataset.image_channels")
    mlp_ratio: int = 2
    qk_norm: bool = True
    mod_embedding: int = "${len:${vision_modality.view_indices}}"
    use_layer_norm: bool = True
