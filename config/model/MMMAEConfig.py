from dataclasses import dataclass

from omegaconf import II, MISSING, OmegaConf, SI

from config.model.UtilsConfig import AnnealingModuleConfig, SigmoidAnnealingModuleConfig

OmegaConf.register_new_resolver("eval", eval)
from typing import Optional

from config.data.DatasetConfig import TextModConfig, VisionModConfig
from config.model.ModelConfig import ModelConfig, ModelModuleConfig_MAE
from config.networks.NetworkConfig import TokenizerModuleConfig


@dataclass
class MAEModelConfig(ModelConfig):
    name: str = MISSING
    patch_size: int = 4
    mask_ratio: float = 0.75
    path: str = "mae.pt"


@dataclass
class MMMAEModuleConfig(ModelModuleConfig_MAE):
    _target_: str = MISSING
    vision_modality_config: Optional[VisionModConfig] = None
    text_modality_config: Optional[TextModConfig] = None
    mask_ratio: float = II("model.mask_ratio")
    learning_rate_weight_decay: float = II("training.weight_decay")
    batch_size: int = II("training.batch_size")
    num_nodes: int = II("training.num_nodes")
    num_gpus: int = II("training.num_gpus")
    n_epochs: int = II("training.n_epochs")
    n_epochs_warmup: int = II("training.n_epochs_warmup")
    sync_logs_on_epoch_end: bool = II("log.sync_logs_on_epoch_end")
    log_debug: bool = II("log.debug")
    encoders = MISSING
    decoders = MISSING
    evaluator = MISSING
    annealingmodule: Optional[AnnealingModuleConfig] = SI(
        "${oc.select:annealingmodule, null}"
    )
    vision_to_text_decoder = MISSING
    vision_to_text_decoder_module = MISSING


@dataclass
class MAEIndependentModelConfig(MAEModelConfig):
    name: str = "independent"
    compute_rec: bool = True


@dataclass
class MAEIndependentModelModuleConfig(MMMAEModuleConfig):
    _target_: str = "mm_maes.mm_mae_independent.MMindependentMAE"
    compute_rec: bool = II("model.compute_rec")


@dataclass
class MAEMMVMModelConfig(MAEModelConfig):
    name: str = "mmvm"
    use_latent_heads: bool = True
    regularization_tokens: str = "all"  # all, all_but_cls, cls
    regularization_metric: str = "cosine_sim"
    pairwise_regularization: bool = True
    compute_rec: bool = True
    temperature: float = 0.5


@dataclass
class MAEMMVMModelModuleConfig(MMMAEModuleConfig):
    _target_: str = "mm_maes.mm_mae_mmvm.MMVMMAE"
    use_latent_heads: bool = II("model.use_latent_heads")
    regularization_tokens: str = II("model.regularization_tokens")
    use_bce_loss: bool = False
    patch_size: int = II("model.patch_size")
    regularization_metric: str = II("model.regularization_metric")
    pairwise_regularization: bool = II("model.pairwise_regularization")
    compute_rec: bool = II("model.compute_rec")
    temperature: float = II("model.temperature")


@dataclass
class MAEAggregationModelConfig(MAEModelConfig):
    name: str = "joint"
    aggregation: str = "avg"
    use_latent_heads: bool = True
    regularization_tokens: str = "all"  # all, all_but_cls, cls


@dataclass
class MAEAggregationModelModuleConfig(MMMAEModuleConfig):
    _target_: str = "mm_maes.mm_mae_joint.MMaggMAE"
    use_latent_heads: bool = II("model.use_latent_heads")
    regularization_tokens: str = II("model.regularization_tokens")
    patch_size: int = II("model.patch_size")


@dataclass
class VisionToTextDecoder:
    text_field: str = "impression"
    loss_weighting_factor: float = 5.0


@dataclass
class VisionToTextDecoderModule:
    _target_: str = MISSING


@dataclass
class SimpleVisionToTextDecoderModule(VisionToTextDecoderModule):
    _target_: str = "networks.NetworksText.SimpleVisionToTextDecoder"
    emb_dim = MISSING


@dataclass
class TransformerVisionToTextDecoderModule(VisionToTextDecoderModule):
    _target_: str = "networks.NetworksText.TFVisionToTextDecoder"
    emb_dim = MISSING
    n_dec_heads: int = 12
    mlp_ratio: int = 4
    qk_norm: bool = True
    n_dec_layers: int = 6
    image_size: int = II("dataset.img_size")
    patch_size: int = II("model.patch_size")


@dataclass
class CapPaVisionToTextDecoderModule(VisionToTextDecoderModule):
    _target_: str = "networks.NetworksText.CapPaVisionToTextDecoder"
    emb_dim = MISSING
    n_dec_heads: int = 8
    mlp_ratio: int = 4
    n_dec_layers: int = 6
    n_img_patches: int = II(
        "eval:'( (${dataset.img_size} // ${model.patch_size}) ** 2 + 1)'"
    )
    cappa_per_view: bool = True
    enable_parallel_decoding: bool = False
    tokenizer_config: TokenizerModuleConfig = II("tokenizermodule")
