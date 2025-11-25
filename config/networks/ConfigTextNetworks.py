from dataclasses import dataclass, field
from omegaconf import MISSING, II


from config.networks.NetworkConfig import TextEncoderModuleConfig
from config.networks.NetworkConfig import DecoderModuleConfig
from config.networks.NetworkConfig import TokenizerModuleConfig


@dataclass
class BiomedBertEncoderModuleConfig(TextEncoderModuleConfig):
    _target_: str = "networks.NetworksText.BiomedBert_Encoder"
    pretrained_model_name: str = (
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    )
    emb_dim: int = 768


@dataclass
class BertEncoderModuleConfig(TextEncoderModuleConfig):
    _target_: str = "networks.NetworksText.BiomedBert_Encoder"
    pretrained_model_name: str = "google-bert/bert-base-uncased"
    emb_dim: int = 768

@dataclass
class BertLargeEncoderModuleConfig(TextEncoderModuleConfig):
    _target_: str = "networks.NetworksText.BiomedBert_Encoder"
    pretrained_model_name: str = "google-bert/bert-large-uncased"
    emb_dim: int = 1024


@dataclass
class BiomedBertDecoderModuleConfig(DecoderModuleConfig):
    _target_: str = "networks.NetworksText.BiomedBert_Decoder"

@dataclass
class BertLargeDecoderModuleConfig(DecoderModuleConfig):
    _target_: str = "networks.NetworksText.BiomedBert_Decoder"
 
@dataclass
class BertLargeTokenizerModuleConfig(TokenizerModuleConfig):
     pretrained_model_name: str = "google-bert/bert-large-uncased"
     mlm_probability: float = 0.15
     max_len: int = 300 #10
     truncation: bool = True

@dataclass
class BertDecoderModuleConfig(DecoderModuleConfig):
    _target_: str = "networks.NetworksText.BiomedBert_Decoder"


@dataclass
class BertLargeDecoderModuleConfig(DecoderModuleConfig):
    _target_: str = "networks.NetworksText.BiomedBert_Decoder"


@dataclass
class BiomedBertTokenizerModuleConfig(TokenizerModuleConfig):
    pretrained_model_name: str = (
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    )
    mlm_probability: float = 0.15
    max_len: int = MISSING
    truncation: bool = True

@dataclass
class BertTokenizerModuleConfig(TokenizerModuleConfig):
    pretrained_model_name: str = "google-bert/bert-base-uncased"
    mlm_probability: float = 0.15
    max_len: int = MISSING
    truncation: bool = True
    dec_token: bool = False
    vlm: bool = False
    pretrained_model_name_vlm: str = ("Henrychur/MMed-Llama-3-8B")