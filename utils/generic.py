import numpy as np
import torch
import pytorch_lightning as pl
from torchmetrics import Metric
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, BatchEncoding
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info
import os

from config.data.DatasetConfig import VisionModConfig, TextModConfig

_SUPPORTED_TEXT_MODALITIES = ["findings", "impression", "impression_plus_findings"]


class EpochPrefixCallback(pl.Callback):
    def __init__(self, prefix):
        self.prefix = prefix

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.logger.log_metrics(
            {f"{self.prefix}/epoch": trainer.current_epoch}, step=trainer.global_step
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        trainer.logger.log_metrics(
            {f"{self.prefix}/val_epoch": trainer.current_epoch},
            step=trainer.global_step,
        )


def check_cfg_logic(cfg):
    # Config logic checks
    if not cfg.training.do_pretraining and not cfg.offline_eval.do_offline_eval:
        raise ValueError(
            "Configuration error: Both 'do_pretraining' and 'do_offline_eval' are set to False.\n"
            "At least one of these options must be True to run the experiment."
        )


class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __call__(self, features, **kwargs):
        if not all(isinstance(feature, tuple) for feature in features):
            raise ValueError("Expected features to be a list of tuples")

        samples_list, labels_list = zip(*features)

        data = {}
        if "text" in samples_list[0].keys():

            data_text = {}
            for t_key in samples_list[0]["text"].keys():
                data_text[t_key] = {}
                text_features = [sample["text"][t_key]["data"] for sample in samples_list]
                collated_text = super().__call__(text_features, **kwargs)
                data_text[t_key]["data"] = collated_text
                text_masks = torch.as_tensor(np.asarray([sample["text"][t_key]["mask"] for
                                            sample in samples_list]))
                data_text[t_key]["mask"] = text_masks
                text_indices = torch.as_tensor(np.asarray([sample["text"][t_key]["index"] for
                                            sample in samples_list]))
                data_text[t_key]["index"] = text_indices
            data["text"] = data_text

        # vision part
        if "vision" in samples_list[0].keys():
            data_vision = {}
            for v_key in samples_list[0]["vision"].keys():
                data_vision[v_key] = {}
                vis_feats = [sample["vision"][v_key]["data"] for sample in samples_list]
                data_vision[v_key]["data"] = torch.stack(vis_feats)
                vis_masks = torch.as_tensor(np.asarray([sample["vision"][v_key]["mask"] for
                                            sample in samples_list]))
                data_vision[v_key]["mask"] = vis_masks
                vis_indices = torch.as_tensor(np.asarray([sample["vision"][v_key]["index"]
                                            for sample in samples_list]))
                data_vision[v_key]["index"] = vis_indices
            data["vision"] = data_vision
        return [data, torch.stack(labels_list)]

def get_data_loaders(
    text_modality_config,
    train_dst,
    val_dst_list,
    batch_size,
    num_workers,
    vision_to_text_reconstruction,
    tokenizer_module,
):
    if text_modality_config is not None or vision_to_text_reconstruction is not None:
        train_loader = torch.utils.data.DataLoader(
            train_dst,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=True,
            collate_fn=CustomDataCollatorForLanguageModeling(
                tokenizer=AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=tokenizer_module.pretrained_model_name
                ),
                mlm=True,
                mlm_probability=tokenizer_module.mlm_probability,
            ),
        )
        val_loader_list = []
        for val_dst in val_dst_list:
            val_loader_list.append(
                torch.utils.data.DataLoader(
                    val_dst,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=False,
                    prefetch_factor=4,
                    drop_last=False,
                    collate_fn=CustomDataCollatorForLanguageModeling(
                        tokenizer=AutoTokenizer.from_pretrained(
                            pretrained_model_name_or_path=tokenizer_module.pretrained_model_name
                        ),
                        mlm=True,
                        mlm_probability=tokenizer_module.mlm_probability,
                    ),
                )
            )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dst,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            drop_last=True,
        )
        val_loader_list = []
        for val_dst in val_dst_list:
            val_loader_list.append(
                torch.utils.data.DataLoader(
                    val_dst,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=False,
                    prefetch_factor=4,
                    drop_last=False,
                )
            )

    return train_loader, val_loader_list


def move_tensors_to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, dict):
        return {k: move_tensors_to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_cpu(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_tensors_to_cpu(item) for item in data)
    elif isinstance(data, set):
        return {move_tensors_to_cpu(item) for item in data}
    else:
        return data

def move_tensors_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, BatchEncoding):
        # special case for text
        return BatchEncoding({k: move_tensors_to_device(v, device) for k, v in data.items()})
    elif isinstance(data, dict):
        return {k: move_tensors_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_tensors_to_device(item, device) for item in data)
    elif isinstance(data, set):
        return {move_tensors_to_device(item, device) for item in data}
    else:
        return data


def get_supported_text_modalities():
    # LIST THAT DEFINES THE MODALITIES OF THE TEXT DOMAIN ACCEPTED BY THE MODEL
    # MODALITIES WITH DIFFERENT NAMES ARE NOT PROCESSED AS TEXT
    return _SUPPORTED_TEXT_MODALITIES.copy()


def has_text_modality(modality_names):
    return any(modality in _SUPPORTED_TEXT_MODALITIES for modality in modality_names)


def is_text_modality(modality):
    return any(m == modality for m in _SUPPORTED_TEXT_MODALITIES)


def get_text_modalities(text_modality_config: TextModConfig,
                        vision_modality_config: VisionModConfig):
    list_text_mods = []
    if text_modality_config is not None:
        list_text_mods = [
            value for value in text_modality_config.views if value in _SUPPORTED_TEXT_MODALITIES
        ]
    if vision_modality_config is not None:
        [list_text_mods.append(value) for value in
         vision_modality_config.decode if value in _SUPPORTED_TEXT_MODALITIES]
    return list(set(list_text_mods))


def remove_text_modalities_from_list(modality_names):
    for tm in _SUPPORTED_TEXT_MODALITIES:
        if tm in modality_names:
            modality_names.remove(tm)


class WandbCheckpointCallback(Callback):
    def __init__(self, wandb_logger, monitor="val_loss", mode="min",
                 checkpoint_freq=1):
        self.wandb_logger = wandb_logger
        self.monitor = monitor
        self.mode = mode
        self._checkpoint_callback = None  # defer construction
        self.checkpoint_freq = checkpoint_freq

    @rank_zero_only
    def _get_checkpoint_dir(self):
        exp = self.wandb_logger.experiment  # triggers wandb.init() safely on rank 0
        dir_attr = exp.dir() if callable(exp.dir) else exp.dir
        checkpoint_dir = os.path.join(dir_attr, "checkpoints")
        rank_zero_info(f"Using checkpoint dir: {checkpoint_dir}")
        return checkpoint_dir

    def on_fit_start(self, trainer, pl_module):
        checkpoint_dir = self._get_checkpoint_dir()

        # broadcast to all ranks
        if torch.distributed.is_initialized():
            dir_list = [checkpoint_dir]
            torch.distributed.broadcast_object_list(dir_list, src=0)
            checkpoint_dir = dir_list[0]

        # create and register actual checkpoint callback
        self._checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=self.monitor,
            mode=self.mode,
            every_n_epochs=self.checkpoint_freq,
            save_last=True,
            save_on_train_epoch_end=False,
        )
        trainer.callbacks.append(self._checkpoint_callback)


class MultilabelBrierScore(Metric):
    """
    Computes Brier score for multilabel classification.

    preds: probabilities in [0,1], shape [N, L]
    target: binary labels {0,1}, shape [N, L]
    """
    is_differentiable = False
    full_state_update = False  # to support distributed

    def __init__(self, num_labels: int, average: str | None = "macro"):
        super().__init__()
        assert average in (None, "macro", "micro"), "average must be None, 'macro' or 'micro'"
        self.num_labels = num_labels
        self.average = average

        # Sum of squared errors per label
        self.add_state("sum_sqerr_per_label", default=torch.zeros(num_labels), dist_reduce_fx="sum")
        # Number of samples seen
        self.add_state("n_samples", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        sq = (preds - target.float()) ** 2  # [N, L]
        self.sum_sqerr_per_label += sq.sum(dim=0)  # sum over batch dimension
        self.n_samples += preds.shape[0]

    def compute(self):
        per_label = self.sum_sqerr_per_label / self.n_samples  # mean per label
        if self.average == "macro":
            return per_label.mean()
        else:
            return per_label