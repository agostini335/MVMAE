import warnings

import torch.nn as nn
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from config.data.ConfigMimic import (
    CXREvalModuleConfig,
    CXRTestModuleConfig,
    CXRTextModConfig,
    CXRTrainModuleConfig,
    CXRVisionAllModConfig,
    CXRVisionFLModConfig,
    CXRVisionFLUModConfig,
    MimicCXRDataConfig,
)
from config.data.DatasetConfig import TextModConfig, VisionModConfig
from config.data.TransformConfig import (
    TransformMimicFolderBiomedClipEvalModule,
    TransformMimicFolderBiomedClipTrainModule,
    TransformMimicFolderMAEEvalModule,
    TransformMimicFolderMAETrainModule,
    TransformMimicFolderStanfordViTEvalModule,
    TransformMimicFolderStanfordViTTrainModule,
)
from config.eval.MimicEvalConfig import (
    MimicBiomedClipEnsembleClf,
    MimicBiomedClipUnimodalClf,
    MimicEnsemblelClf,
    MimicUnimodalClf,
    MimicEvalConfig,
    MimicEvalOfflineConfig,
    MimicMAEEvalModuleConfig,
    MimicStanfordViTEnsembleClf,
    MimicStanfordViTUnimodalClf,
) 
from config.model.MMMAEConfig import (
    CapPaVisionToTextDecoderModule,
    MAEAggregationModelConfig,
    MAEAggregationModelModuleConfig,
    MAEIndependentModelConfig,
    MAEIndependentModelModuleConfig,
    MAEMMVMModelConfig,
    MAEMMVMModelModuleConfig,
    SimpleVisionToTextDecoderModule,
    TransformerVisionToTextDecoderModule,
    VisionToTextDecoder,
)
from config.model.UtilsConfig import (
    CosAnnealingConfig,
    CosAnnealingModuleConfig,
    ExpAnnealingConfig,
    ExpAnnealingModuleConfig,
    LinearAnnealingConfig,
    LinearAnnealingModuleConfig,
    NoAnnealingConfig,
    NoAnnealingModuleConfig,
    SigmoidAnnealingConfig,
    SigmoidAnnealingModuleConfig,
)
from config.MyMVWSLConfig import (
    LogConfig,
    MAETrainingConfig,
    MyMVWSLConfig,
    TrainingConfig,
)
from config.networks.ConfigMAENetworks import (
    ViTbaseDecoderModuleConfig,
    ViTbaseEncoderModuleConfig,
    ViTlargeDecoderModuleConfig,
    ViTlargeEncoderModuleConfig,
    ViTtinyDecoderModuleConfig,
    ViTtinyEncoderModuleConfig,
)
from config.networks.ConfigTextNetworks import (
    BertDecoderModuleConfig,
    BertEncoderModuleConfig,
    BertLargeDecoderModuleConfig,
    BertLargeEncoderModuleConfig,
    BertLargeTokenizerModuleConfig,
    BertTokenizerModuleConfig,
    BiomedBertDecoderModuleConfig,
    BiomedBertEncoderModuleConfig,
    BiomedBertTokenizerModuleConfig,
)
from config.networks.NetworkConfig import DecoderModuleConfig, EncoderModuleConfig


def resolve_clf(ft_strategy: str) -> str:
    mapping = {
        "frontal": "classifiers.Classifiers.CLFTeset",
        "lateral": "classifiers.Classifiers.CLFTeset",
        "ensemble": "classifiers.Classifiers.CLFTeset",
    }
    if ft_strategy not in mapping:
        raise ValueError(f"Unknown strategy: {ft_strategy}")
    return mapping[ft_strategy]


def get_encoders(
    vision_encoder: EncoderModuleConfig,
    text_encoder: EncoderModuleConfig,
    vision_modality_config: VisionModConfig,
    text_modality_config: TextModConfig,
    compute_rec: bool = True,
) -> nn.ModuleDict:
    enc_modules = nn.ModuleDict()

    if text_modality_config is not None:
        enc_modules.add_module(
            "text", instantiate(text_encoder, compute_rec=compute_rec)
        )
    else:
        warnings.warn(
            "Instantiating encoder modules without text modality", UserWarning
        )
    if vision_modality_config is not None:
        enc_modules.add_module("vision", instantiate(vision_encoder))
    else:
        warnings.warn(
            "Instantiating encoder modules without vision modality", UserWarning
        )
    return enc_modules


def get_decoders(
    vision_decoder: DecoderModuleConfig,
    text_decoder: DecoderModuleConfig,
    vision_modality_config: VisionModConfig,
    text_modality_config: TextModConfig,
) -> nn.ModuleDict:
    dec_modules = nn.ModuleDict()

    if text_modality_config is not None:
        dec_modules.add_module("text", instantiate(text_decoder))
        dec_modules.add_module("text", instantiate(text_decoder))
    else:
        warnings.warn(
            "Instantiating decoder modules without text modality", UserWarning
        )
    if vision_modality_config is not None:
        dec_modules.add_module("vision", instantiate(vision_decoder))
    else:
        warnings.warn(
            "Instantiating decoder modules without vision modality", UserWarning
        )
    return dec_modules


def load_config():
    cs = ConfigStore.instance()
    cs.store(group="log", name="log", node=LogConfig)
    cs.store(group="model", name="mae_independent", node=MAEIndependentModelConfig)
    cs.store(group="model", name="mae_mmvm", node=MAEMMVMModelConfig)
    cs.store(group="model", name="mae_joint", node=MAEAggregationModelConfig)

    cs.store(
        group="modelmodule", name="mae_agg_module", node=MAEAggregationModelModuleConfig
    )
    cs.store(
        group="modelmodule",
        name="mae_independent_module",
        node=MAEIndependentModelModuleConfig,
    )
    cs.store(group="modelmodule", name="mae_mmvm_module", node=MAEMMVMModelModuleConfig)
    cs.store(
        group="encodermodule",
        name="vit_tiny_encoder_module",
        node=ViTtinyEncoderModuleConfig,
    )
    cs.store(
        group="decodermodule",
        name="vit_tiny_decoder_module",
        node=ViTtinyDecoderModuleConfig,
    )
    cs.store(
        group="encodermodule",
        name="vit_base_encoder_module",
        node=ViTbaseEncoderModuleConfig,
    )
    cs.store(
        group="decodermodule",
        name="vit_base_decoder_module",
        node=ViTbaseDecoderModuleConfig,
    )
    cs.store(
        group="encodermodule",
        name="vit_large_encoder_module",
        node=ViTlargeEncoderModuleConfig,
    )
    cs.store(
        group="decodermodule",
        name="vit_large_decoder_module",
        node=ViTlargeDecoderModuleConfig,
    )
    cs.store(
        group="textencodermodule",
        name="biomed_bert_encoder_module",
        node=BiomedBertEncoderModuleConfig,
    )
    cs.store(
        group="textdecodermodule",
        name="biomed_bert_decoder_module",
        node=BiomedBertDecoderModuleConfig,
    )
    cs.store(
        group="tokenizermodule",
        name="biomed_bert_tokenizer_module",
        node=BiomedBertTokenizerModuleConfig,
    )
    cs.store(
        group="textencodermodule",
        name="bert_encoder_module",
        node=BertEncoderModuleConfig,
    )
    cs.store(
        group="textdecodermodule",
        name="bert_decoder_module",
        node=BertDecoderModuleConfig,
    )
    cs.store(
        group="tokenizermodule",
        name="bert_tokenizer_module",
        node=BertTokenizerModuleConfig,
    )
    cs.store(
        group="textencodermodule",
        name="bert_large_encoder_module",
        node=BertLargeEncoderModuleConfig,
    )
    cs.store(
        group="textdecodermodule",
        name="bert_large_decoder_module",
        node=BertLargeDecoderModuleConfig,
    )
    cs.store(
        group="tokenizermodule",
        name="bert_large_tokenizer_module",
        node=BertLargeTokenizerModuleConfig,
    )
    cs.store(group="annealing", name="sigmoid", node=SigmoidAnnealingConfig)
    cs.store(group="annealing", name="cosine", node=CosAnnealingConfig)
    cs.store(group="annealing", name="exp", node=ExpAnnealingConfig)
    cs.store(group="annealing", name="linear", node=LinearAnnealingConfig)
    cs.store(group="annealing", name="noannealing", node=NoAnnealingConfig)
    cs.store(
        group="annealingmodule", name="sigmoidmodule", node=SigmoidAnnealingModuleConfig
    )
    cs.store(group="annealingmodule", name="cosmodule", node=CosAnnealingModuleConfig)
    cs.store(group="annealingmodule", name="expmodule", node=ExpAnnealingModuleConfig)
    cs.store(
        group="annealingmodule", name="linearmodule", node=LinearAnnealingModuleConfig
    )
    cs.store(
        group="annealingmodule", name="noannealingmodule", node=NoAnnealingModuleConfig
    )
    cs.store(group="training", name="training", node=TrainingConfig)
    cs.store(group="training", name="training_mae", node=MAETrainingConfig)

    cs.store(group="eval", name="MimicEval", node=MimicEvalConfig)
    cs.store(group="offline_eval", name="MimicOfflineEval", node=MimicEvalOfflineConfig)

    cs.store(
        group="offline_evalmodule", name="mimic_ensemble_clf", node=MimicEnsemblelClf
    )
    cs.store(
        group="offline_evalmodule", name="mimic_unimodal_clf", node=MimicUnimodalClf
    )
    cs.store(
        group="offline_evalmodule",
        name="mimic_biomedclip_ensemble_clf",
        node=MimicBiomedClipEnsembleClf,
    )
    cs.store(
        group="offline_evalmodule",
        name="mimic_biomedclip_unimodal_clf",
        node=MimicBiomedClipUnimodalClf,
    )
    cs.store(
        group="offline_evalmodule",
        name="mimic_stanfordvit_ensemble_clf",
        node=MimicStanfordViTEnsembleClf,
    )
    cs.store(
        group="offline_evalmodule",
        name="mimic_stanfordvit_unimodal_clf",
        node=MimicStanfordViTUnimodalClf,
    )
    cs.store(
        group="evalmodule", name="MimicMAEEvalModule", node=MimicMAEEvalModuleConfig
    )
    cs.store(group="dataset", name="Mimic_cxr", node=MimicCXRDataConfig)

    cs.store(
        group="datamodule_train",
        name="CXRTrainDataModule",
        node=CXRTrainModuleConfig,
    )
    cs.store(
        group="datamodule_eval",
        name="CXREvalDataModule",
        node=CXREvalModuleConfig,
    )
    cs.store(
        group="datamodule_test",
        name="CXRTestDataModule",
        node=CXRTestModuleConfig,
    )

    cs.store(
        group="transformmodule_train",
        name="transformmodule_mimic_train",
        node=TransformMimicFolderMAETrainModule,
    )
    cs.store(
        group="transformmodule_eval",
        name="transformmodule_mimic_eval",
        node=TransformMimicFolderMAEEvalModule,
    )
    cs.store(
        group="transformmodule_train",
        name="transformmodule_biomedclip_mimic_train",
        node=TransformMimicFolderBiomedClipTrainModule,
    )
    cs.store(
        group="transformmodule_eval",
        name="transformmodule_biomedclip_mimic_eval",
        node=TransformMimicFolderBiomedClipEvalModule,
    )
    cs.store(
        group="transformmodule_train",
        name="transformmodule_stanfordvit_mimic_train",
        node=TransformMimicFolderStanfordViTTrainModule,
    )
    cs.store(
        group="transformmodule_eval",
        name="transformmodule_stanfordvit_mimic_eval",
        node=TransformMimicFolderStanfordViTEvalModule,
    )
    cs.store(
        group="vision_to_text_decoder_module",
        name="simple_vtt_decoder",
        node=SimpleVisionToTextDecoderModule,
    )
    cs.store(
        group="vision_to_text_decoder_module",
        name="tf_vtt_decoder",
        node=TransformerVisionToTextDecoderModule,
    )
    cs.store(
        group="vision_to_text_decoder_module",
        name="cappa_decoder",
        node=CapPaVisionToTextDecoderModule,
    )
    cs.store(
        group="vision_to_text_decoder",
        name="vtt_decoder",
        node=VisionToTextDecoder,
    )
    cs.store(
        group="vision_modality",
        name="cxr_vision_mod_fl",
        node=CXRVisionFLModConfig,
    )
    cs.store(
        group="vision_modality",
        name="cxr_vision_mod_flu",
        node=CXRVisionFLUModConfig,
    )
    cs.store(
        group="vision_modality",
        name="cxr_vision_mod_all",
        node=CXRVisionAllModConfig,
    )
    cs.store(
        group="text_modality",
        name="cxr_text_mod",
        node=CXRTextModConfig,
    )
    cs.store(name="base_config", node=MyMVWSLConfig)
