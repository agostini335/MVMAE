from dataclasses import dataclass, field
from typing import Any, List, Optional
from omegaconf import MISSING
from omegaconf import II
from config.eval.EvalConfig import EvalConfig, OfflineEvalConfig
from config.eval.EvalConfig import EvalModuleConfig, OfflineEvalModuleConfig
from config.data.ConfigMimic import CXRVisionModConfig, CXRTextModConfig


@dataclass
class MimicEvalBaseConfig(EvalConfig):
    # validation datasets
    validation_dataset_names: List[str] = field(
        default_factory=lambda: [
            "mimic-cxr",
            # "rexgradient",
            "chexpertplus",
            "padchest",
            # "chest-xray",
        ]
    )
    # If True, adds an additional joint validation dataset containing all validation_dataset_names
    add_joint_validation_dataset: bool = True
    evaluate_coherence: bool = False

    classifier_list: List[str] = field(
        default_factory=lambda: [
            "LR",
        ]
    )

    metric_list: List[str] = field(
        default_factory=lambda: [
            "AUROC",
            "AP",
            "F1",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
        ]
    )
    f_n_jobs: int = 8


@dataclass
class MimicEvalBaseModuleConfig(EvalModuleConfig):
    _target_: str = MISSING
    vision_modality_config: Optional[CXRVisionModConfig] = None
    text_modality_config: Optional[CXRTextModConfig] = None
    clfs: List[str] = II("eval.classifier_list")
    metrics: List[str] = II("eval.metric_list")
    f_n_jobs: int = II("eval.f_n_jobs")
    seed: int = II("training.seed")


@dataclass
class MimicEvalConfig(MimicEvalBaseConfig):
    # RF PARAMETERS
    f_n_estimators: int = 1000
    f_min_samples_split: int = 5
    f_min_samples_leaf: int = 1
    f_max_features: str = "sqrt"
    f_max_depth: int = 5
    f_criterion: str = "entropy"
    f_bootstrap: bool = True


@dataclass
class MimicMAEEvalModuleConfig(MimicEvalBaseModuleConfig):
    _target_: str = "eval.MimicEval.MimicMAEEval"
    f_n_estimators: int = II("eval.f_n_estimators")
    f_min_samples_split: int = II("eval.f_min_samples_split")
    f_min_samples_leaf: int = II("eval.f_min_samples_leaf")
    f_max_features: str = II("eval.f_max_features")
    f_max_depth: int = II("eval.f_max_depth")
    f_criterion: str = II("eval.f_criterion")
    f_bootstrap: bool = II("eval.f_bootstrap")


@dataclass
class MimicEvalOfflineConfig(OfflineEvalConfig):
    do_offline_eval: bool = True
    # model checkpoint path to evaluate
    ckpt_path: str = ""

    # Fine Tuning Pipeline Params
    freeze_encoder: bool = (
        False  # if 'True', encoders is freezed (probing). If 'False' full fine tuning is performed.
    )
    clf_head_type: str = "linear"  #'linear' or 'non_linear' classifier head
    feature_policy: str = "cls"  # only cls is implemented
    checkpoint_init: bool = (
        True  # False for supervised clf baseline - the checkpoint is not loaded
    )

    # training setup
    oe_auroc_frequency: int = 1  # auroc is logged every oe_auroc_frequency epochs


@dataclass
class MimicBaseClf(OfflineEvalModuleConfig):
    _target_: str = MISSING
    oe_encoders: Optional[Any] = None
    oe_batch_size: int = II("training.batch_size")
    oe_num_nodes: int = II("training.num_nodes")
    oe_num_gpus: int = II("training.num_gpus")
    oe_num_labels: int = II("dataset.num_labels")

    oe_n_epochs: int = 100
    oe_learning_rate: float = 5e-5

    oe_learning_rate_weight_decay: float = II("training.weight_decay")
    oe_checkpoint_init: bool = II("offline_eval.checkpoint_init")
    oe_freeze_encoder: bool = II("offline_eval.freeze_encoder")
    oe_feature_policy: str = II("offline_eval.feature_policy")
    oe_ckpt_path: str = II("offline_eval.ckpt_path")
    oe_clf_head_type: str = II("offline_eval.clf_head_type")
    oe_logging_frequency: int = II("log.wandb_log_freq")

    oe_auroc_frequency: int = II("offline_eval.oe_auroc_frequency")
    oe_n_epochs_warmup: int = II("training.n_epochs_warmup")

    # vision modality max views
    oe_max_num_views: int = II("vision_modality.max_num_views")


@dataclass
class MimicEnsemblelClf(MimicBaseClf):
    _target_: str = "classifiers.Classifiers.EnsembleClassifier"


@dataclass
class MimicUnimodalClf(MimicBaseClf):
    _target_: str = "classifiers.Classifiers.UnimodalClassifier"


@dataclass
class MimicBiomedClipEnsembleClf(MimicBaseClf):
    _target_: str = "classifiers.Classifiers.BiomedClipEnsembleClassifier"
    cache_dir: str = "cache_dir"


@dataclass
class MimicBiomedClipUnimodalClf(MimicBaseClf):
    _target_: str = "classifiers.Classifiers.BiomedClipUnimodalClassifier"
    cache_dir: str = "cache_dir"


@dataclass
class MimicStanfordViTEnsembleClf(MimicBaseClf):
    _target_: str = "classifiers.Classifiers.StanfordViTEnsembleClassifier"
    cache_dir: str = "cache_dir"


@dataclass
class MimicStanfordViTUnimodalClf(MimicBaseClf):
    _target_: str = "classifiers.Classifiers.StanfordViTUnimodalClassifier"
    cache_dir: str = "cache_dir"
