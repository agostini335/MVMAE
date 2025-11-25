import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from omegaconf import MISSING
from omegaconf import II

from config.data.DatasetConfig import DataConfig
from config.data.DatasetConfig import TrainDataModuleConfig
from config.data.DatasetConfig import EvalDataModuleConfig
from config.data.DatasetConfig import TestDataModuleConfig
from config.data.DatasetConfig import VisionModConfig
from config.data.DatasetConfig import TextModConfig


@dataclass
class CXRVisionModConfig(VisionModConfig):
    max_num_views: int = 2
    views: List[str] = field(default_factory=lambda: [])
    decode: List[str] = field(default_factory=lambda: [])
    view_indices: Dict[str, int] = field(default_factory=lambda: {})


@dataclass
class CXRVisionFLModConfig(CXRVisionModConfig):
    max_num_views: int = 2
    views: List[str] = field(default_factory=lambda: ["F", "L"])
    # impression works if we have vision to text decoder - put only self for std rec loss
    decode: List[str] = field(default_factory=lambda: ["self", "impression"])
    view_indices: Dict[str, int] = field(default_factory=lambda: {"F": 0, "L": 1})


@dataclass
class CXRVisionFLUModConfig(CXRVisionModConfig):
    max_num_views: int = 3
    views: List[str] = field(default_factory=lambda: ["F", "L", "U"])
    # decode: List[str] = field(default_factory=lambda: ["self", "impression"])
    decode: List[str] = field(default_factory=lambda: ["self"])
    view_indices: Dict[str, int] = field(
        default_factory=lambda: {"F": 0, "L": 1, "U": 2}
    )


@dataclass
class CXRVisionAllModConfig(CXRVisionModConfig):
    max_num_views: int = 3
    views: List[str] = field(default_factory=lambda: ["all"])
    decode: List[str] = field(default_factory=lambda: ["self", "impression"])
    view_indices: Dict[str, int] = field(
        default_factory=lambda: {
            "AP": 0,
            "AP AXIAL": 1,
            "AP RLD": 2,
            "AP LLD": 3,
            "PA": 4,
            "PA LLD": 5,
            "PA RLD": 6,
            "LATERAL": 7,
            "LL": 8,
            "RL": 9,
            "LAO": 10,
            "LPO": 11,
            "RAO": 12,
            "RPO": 13,
            "OBLIQUE": 14,
            "SUPINE": 15,
            "ERECT": 16,
            "LLD": 17,
            "DECUBITUS": 18,
            "XTABLE LATERAL": 19,
            "SWIMMERS": 20,
            "KUB": 21,
            "PICC LINE": 22,
            "GENERICA": 23,
            "UNKNOWN": 24,
        }
    )


@dataclass
class CXRTextModConfig(TextModConfig):
    max_num_views: int = 2
    views: List[str] = field(default_factory=lambda: ["impression", "findings"])
    decode: List[str] = field(
        default_factory=lambda: [
            "self",
        ]
    )
    view_indices: Dict[str, int] = field(
        default_factory=lambda: {"impression": 0, "findings": 1}
    )

@dataclass
class MimicCXRDataConfig(DataConfig):
    name: str = "mm dataset"
    dir_data: str = "dir_data"

    # metadata config
    split_seed: int = 100
    metadata_img_size: int = 256
    metadata_version: str = "1.0"

    # img config
    img_size: int = 224
    image_channels: int = 1

    # study settings
    studies_policy: str = "all_combi_no_missing"
    reduced_dataset: bool = False
    num_train_sample: int = 20000

    # view modality settings
    ref_mod_d_size: int = 50176  # img-size * img-size = 224*224
    modalities_size: Dict[str, int] = field(
        default_factory=lambda: {"frontal": 50176, "lateral": 50176}
    )

    # labels
    target_list: List[str] = field(
        default_factory=lambda: [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Lung Opacity",
            "No Finding",
            "Pleural Effusion",
            "Pleural Other",
            "Pneumonia",
            "Pneumothorax",
            "Support Devices",
        ]
    )
    num_labels: int = 14


@dataclass
class CXRTrainModuleConfig(TrainDataModuleConfig):
    _target_: str = "datasets.MimicCXRDataset.CXRDataset"
    vision_modality_config: Optional[VisionModConfig] = None
    text_modality_config: Optional[CXRTextModConfig] = None
    dir_data: str = II("dataset.dir_data")
    split: str = "train"
    img_size: int = II("dataset.img_size")
    target_list: List[str] = II("dataset.target_list")
    debug: bool = II("dataset.reduced_dataset")
    num_train_sample: int = II("dataset.num_train_sample")
    split_seed: int = II("dataset.split_seed")
    version: str = II("dataset.metadata_version")
    studies_policy: str = II("dataset.studies_policy")
    fn_metadata: str = os.path.join(
        dir_data,
        "metadata",
        f"metadata_seed{II('dataset.split_seed')}_train_{II('dataset.metadata_img_size')}_v{II('dataset.metadata_version')}.csv",
    )
    vision_text_reconstruction = MISSING
    transform = MISSING
    tokenizer = MISSING
    selected_datasets: List[str] = field(
        default_factory=lambda: [
            "mimic-cxr",
            "chexpertplus",
            "padchest",
            "chest-xray",
        ]
    )


@dataclass
class CXREvalModuleConfig(EvalDataModuleConfig):
    _target_: str = "datasets.MimicCXRDataset.CXRDataset"
    vision_modality_config: Optional[VisionModConfig] = None
    text_modality_config: Optional[CXRTextModConfig] = None
    split: str = "val"
    dir_data: str = II("dataset.dir_data")
    img_size: int = II("dataset.img_size")
    target_list: List[str] = II("dataset.target_list")
    debug: bool = False
    num_train_sample: int = 0
    split_seed: int = II("dataset.split_seed")
    version: str = II("dataset.metadata_version")
    studies_policy: str = II("dataset.studies_policy")
    fn_metadata: str = os.path.join(
        dir_data,
        "metadata",
        f"metadata_seed{II('dataset.split_seed')}_val_{II('dataset.metadata_img_size')}_v{II('dataset.metadata_version')}.csv",
    )
    vision_text_reconstruction = MISSING
    transform = MISSING
    tokenizer = MISSING


@dataclass
class CXRTestModuleConfig(TestDataModuleConfig):
    _target_: str = "datasets.MimicCXRDataset.CXRDataset"
    vision_modality_config: Optional[VisionModConfig] = None
    text_modality_config: Optional[CXRTextModConfig] = None
    dir_data: str = II("dataset.dir_data")
    split: str = "test"
    fn_metadata: str = os.path.join(
        dir_data,
        "metadata",
        f"metadata_seed{II('dataset.split_seed')}_test_{II('dataset.metadata_img_size')}_v{II('dataset.metadata_version')}.csv",
    )
    img_size: int = II("dataset.img_size")
    target_list: List[str] = II("dataset.target_list")
    debug: bool = False
    num_train_sample: int = 0
    split_seed: int = II("dataset.split_seed")
    version: str = II("dataset.metadata_version")
    studies_policy: str = II("dataset.studies_policy")
    vision_text_reconstruction = MISSING
    transform = MISSING
    tokenizer = MISSING
