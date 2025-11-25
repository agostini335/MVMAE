import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from omegaconf import MISSING
from omegaconf import II


@dataclass
class DataConfig:
    name: str = MISSING
    num_workers: int = 8
    ref_mod_d_size: int = MISSING
    modalities_size: Dict[str, int] = MISSING
    # path to the data root directory
    dir_data: str = MISSING


@dataclass
class DataModuleConfig:
    _target_: str = MISSING


@dataclass
class TrainDataModuleConfig(DataModuleConfig):
    _target_: str = MISSING
    transform = MISSING


@dataclass
class EvalDataModuleConfig(DataModuleConfig):
    _target_: str = MISSING
    transform = MISSING


@dataclass
class TestDataModuleConfig(DataModuleConfig):
    _target_: str = MISSING
    transform = MISSING

@dataclass
class ModConfig:
    max_num_views: int = MISSING
    views: Optional[List] = None
    decode: Optional[List[Any]] = None


@dataclass
class VisionModConfig(ModConfig):
    max_num_views: int = MISSING
    views: List[str] = field(default_factory=lambda: [])
    decode: Optional[List[str]] = None

@dataclass
class TextModConfig(ModConfig):
    max_num_views: int = MISSING
    views: List[str] = field(default_factory=lambda: [])
    decode: Optional[List[str]] = None
