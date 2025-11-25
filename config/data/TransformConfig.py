from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from torchvision import transforms

from omegaconf import MISSING
from omegaconf import II


@dataclass
class TransformModuleConfig:
    _target_: str = MISSING


@dataclass
class TransformMimicFolderMAETrainModule(TransformModuleConfig):
    _target_: str = "datasets.MimicCXRDataset.get_transform_folder_mae"
    img_size: int = II("dataset.img_size")
    resize_crop: bool = True
    resize_crop_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            size=II("dataset.img_size"),
            scale=(0.5, 1.0),
            ratio=(1.0, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
    )
    rotation: bool = False
    rotation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            degrees=(0, 45),
            interpolation=transforms.InterpolationMode.NEAREST,
            expand=False,
            center=None,
            fill=0,
        )
    )
    horizontal_flip: bool = True
    normalize: bool = True
    normalize_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: dict(mean=[0.5056], std=[0.252])
    )
    jitter: bool = True
    jitter_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: dict(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
    )
    rgb: bool = False


@dataclass
class TransformMimicFolderMAEEvalModule(TransformModuleConfig):
    _target_: str = "datasets.MimicCXRDataset.get_transform_folder_mae"
    img_size: int = II("dataset.img_size")
    resize_crop: bool = False
    resize_crop_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            size=II("dataset.img_size"),
            scale=(0.8, 1.0),
            ratio=(1.0, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
    )
    rotation: bool = False
    rotation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            degrees=(0, 45),
            interpolation=transforms.InterpolationMode.NEAREST,
            expand=False,
            center=None,
            fill=0,
        )
    )
    horizontal_flip: bool = False
    normalize: bool = True
    normalize_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: dict(mean=[0.5056], std=[0.252])
    )
    jitter: bool = False
    jitter_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: dict(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
    )
    rgb: bool = False



@dataclass
class TransformMimicFolderBiomedClipTrainModule(TransformModuleConfig):
    """
    We want to replicate the transformations used in the biomedclip processor:

        Compose(
            Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
            CenterCrop(size=(224, 224))
            <function _convert_to_rgb at 0x400308a66050>
            ToTensor()
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        )

    """

    _target_: str = "datasets.MimicCXRDataset.get_transform_folder_mae"
    img_size: int = II("dataset.img_size")
    resize_crop: bool = False
    rotation: bool = False
    horizontal_flip: bool = False
    normalize: bool = True
    normalize_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: dict(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
    )
    jitter: bool = False
    rgb: bool = True

    # not used
    resize_crop_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            size=II("dataset.img_size"),
            scale=(0.8, 1.0),
            ratio=(1.0, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
    )
    rotation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            degrees=(0, 45),
            interpolation=transforms.InterpolationMode.NEAREST,
            expand=False,
            center=None,
            fill=0,
        )
    )
    jitter_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: dict(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
    )


@dataclass
class TransformMimicFolderBiomedClipEvalModule(TransformModuleConfig):
    """
    We want to replicate the transformations used in the biomedclip processor:

        Compose(
            Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
            CenterCrop(size=(224, 224))
            <function _convert_to_rgb at 0x400308a66050>
            ToTensor()
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        )

    """

    _target_: str = "datasets.MimicCXRDataset.get_transform_folder_mae"
    img_size: int = II("dataset.img_size")
    resize_crop: bool = False
    rotation: bool = False
    horizontal_flip: bool = False
    normalize: bool = True
    normalize_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: dict(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
    )
    jitter: bool = False
    rgb: bool = True

    # not used
    resize_crop_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            size=II("dataset.img_size"),
            scale=(0.8, 1.0),
            ratio=(1.0, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
    )
    rotation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: dict(
            degrees=(0, 45),
            interpolation=transforms.InterpolationMode.NEAREST,
            expand=False,
            center=None,
            fill=0,
        )
    )
    jitter_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: dict(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
    )


@dataclass
class TransformMimicFolderStanfordViTTrainModule(TransformModuleConfig):
    _target_: str = "datasets.MimicCXRDataset.get_transform_folder_stanfordvit"
    cache_dir: str = II("offline_evalmodule.cache_dir")


@dataclass
class TransformMimicFolderStanfordViTEvalModule(TransformModuleConfig):
    _target_: str = "datasets.MimicCXRDataset.get_transform_folder_stanfordvit"
    cache_dir: str = II("offline_evalmodule.cache_dir")
