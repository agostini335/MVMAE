import torch
from torch import nn
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelAveragePrecision,
)
import pytorch_lightning as pl
from torch.cuda.amp import autocast
from transformers import AutoModelForCausalLM
from utils.generic import (
    MultilabelBrierScore,
)


from open_clip import create_model_from_pretrained


class UnimodalClassifier(pl.LightningModule):
    def __init__(
        self,
        oe_batch_size,
        oe_num_nodes,
        oe_num_gpus,
        oe_n_epochs,
        oe_n_epochs_warmup,
        oe_learning_rate,
        oe_learning_rate_weight_decay,
        oe_encoders,
        oe_num_labels: int,
        oe_checkpoint_init: bool,
        oe_freeze_encoder: bool,
        oe_feature_policy: str,
        oe_ckpt_path: str,
        oe_clf_head_type: str,
        oe_logging_frequency: int,
        oe_auroc_frequency: int,
        oe_validation_dataset_names: list,
        oe_latent_attention_module,
        # max num views to consider per study
        oe_max_num_views: int,
    ):
        super().__init__()
        self.oe_max_num_views = oe_max_num_views
        self.oe_num_labels = oe_num_labels
        self.oe_checkpoint_init = oe_checkpoint_init
        self.oe_freeze_encoder = oe_freeze_encoder
        self.oe_feature_policy = oe_feature_policy
        self.oe_ckpt_path = oe_ckpt_path
        self.oe_clf_head_type = oe_clf_head_type
        self.oe_validation_dataset_names = oe_validation_dataset_names
        self.oe_logging_frequency = oe_logging_frequency
        self.oe_batch_size = oe_batch_size
        self.oe_num_nodes = oe_num_nodes
        self.oe_num_gpus = oe_num_gpus
        self.oe_n_epochs = oe_n_epochs
        self.oe_n_epochs_warmup = oe_n_epochs_warmup
        self.oe_base_learning_rate = oe_learning_rate
        self.oe_learning_rate_weight_decay = oe_learning_rate_weight_decay
        self.oe_auroc_frequency = oe_auroc_frequency

        # we save only the vision encoder, we don't use other modality encoders
        self.oe_vis_encoder = oe_encoders["vision"]
        self.save_hyperparameters(ignore=["oe_vis_encoder", "oe_encoders"])
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # CLF head definition
        emb_dim = self.oe_vis_encoder.emb_dim
        if oe_clf_head_type == "non_linear":
            self.clf_head = nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(emb_dim // 2, oe_num_labels),
            )
        elif oe_clf_head_type == "linear":
            self.clf_head = nn.Linear(emb_dim, oe_num_labels)
        else:
            raise Exception("CLF HEAD TYPE NOT SUPPORTED")

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.auroc_dict = nn.ModuleDict(
            {
                name: MultilabelAUROC(num_labels=oe_num_labels, average=None)
                for name in self.oe_validation_dataset_names
            }
        )
        self.auroc_macro_dict = nn.ModuleDict(
            {
                name: MultilabelAUROC(num_labels=oe_num_labels, average="macro")
                for name in self.oe_validation_dataset_names
            }
        )
        self.accuracy_macro_dict = nn.ModuleDict(
            {
                name: MultilabelAccuracy(num_labels=oe_num_labels, average="macro")
                for name in self.oe_validation_dataset_names
            }
        )
        self.f1_dict = nn.ModuleDict(
            {
                name: MultilabelF1Score(num_labels=oe_num_labels, average=None)
                for name in self.oe_validation_dataset_names
            }
        )
        self.f1_macro_dict = nn.ModuleDict(
            {
                name: MultilabelF1Score(num_labels=oe_num_labels, average="macro")
                for name in self.oe_validation_dataset_names
            }
        )
        self.ap_macro_dict = nn.ModuleDict(
            {
                name: MultilabelAveragePrecision(
                    num_labels=oe_num_labels, average="macro"
                )
                for name in self.oe_validation_dataset_names
            }
        )
        self.brier_macro_dict = nn.ModuleDict(
            {
                name: MultilabelBrierScore(num_labels=oe_num_labels, average="macro")
                for name in self.oe_validation_dataset_names
            }
        )

        if self.oe_freeze_encoder:
            print("Freezing vision encoder weights")
            for param in self.oe_vis_encoder.parameters():
                param.requires_grad = False
        else:
            print("Full Fine-tuning of vision encoder")

    def select_features(self, feats):
        # adapted from: select_alignment_tokens(self, feats) MMVMMAE

        if self.oe_feature_policy == "cls":
            # use only CLS token
            lat = feats.movedim(0, 1)[:, 0, :].unsqueeze(1)
        else:
            raise ValueError("oe_feature_policy is implemented only as cls")
        return lat

    def forward(self, batch):
        data, labels = batch
        logits_list = []
        avail_mask_list = []
        enc_out = {}

        self.oe_vis_encoder.set_mask_ratio(0.0)

        for v_key, v_val in data["vision"].items():
            inputs = {}
            inputs["data"] = v_val["data"]
            inputs["modality_index"] = v_val["index"]
            e_m_out = self.oe_vis_encoder(inputs)
            enc_out[f"vision_{v_key}"] = e_m_out
            lat = self.select_features(e_m_out["features"])
            logits = self.clf_head(lat)
            logits_list.append(logits)
            avail_mask_list.append(v_val["mask"])

        logits = torch.stack(
            logits_list, dim=1
        )  # shape: (batch_size, num_views, 1, num_labels)
        avail_mask = (
            torch.stack(avail_mask_list, dim=1).unsqueeze(-1).unsqueeze(-1)
        )  # shape: (batch_size, num_views, 1, 1)

        return logits, labels, avail_mask

    def training_step(self, batch, batch_idx):
        logits, labels, avail_mask = self.forward(batch)

        # Check if any samples have no available views
        total_views_per_sample = (
            avail_mask.sum(dim=1).squeeze(-1).squeeze(-1)
        )  # shape: (batch_size,)
        valid_samples = total_views_per_sample > 0

        if not valid_samples.any():
            # All samples have no available views - return zero loss
            return torch.tensor(0.0, requires_grad=True, device=logits.device)

        # Filter out samples with no available views
        valid_logits = logits[valid_samples]
        valid_labels = labels[valid_samples]
        valid_avail_mask = avail_mask[valid_samples]

        # Expand labels to match logits shape: (valid_batch_size, num_views, 1, num_labels)
        labels_expanded = valid_labels.unsqueeze(1).unsqueeze(2).expand_as(valid_logits)

        # Calculate loss for each view independently
        view_losses = self.loss_fn(
            valid_logits, labels_expanded
        )  # shape: (valid_batch_size, num_views, 1)

        # Apply availability mask and sum losses only for available views
        masked_losses = view_losses * valid_avail_mask.squeeze(
            -1
        )  # shape: (valid_batch_size, num_views, 1)
        loss = (
            masked_losses.sum() / valid_avail_mask.sum()
        )  # average across all available views

        print("Training loss:", loss)
        self.log("offline_eval/train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits, labels, avail_mask = self.forward(batch)

        # Check if any samples have no available views
        total_views_per_sample = (
            avail_mask.sum(dim=1).squeeze(-1).squeeze(-1)
        )  # shape: (batch_size,)
        valid_samples = total_views_per_sample > 0

        if not valid_samples.any():
            # All samples have no available views - return zero loss and skip metrics
            dataset_name = self.oe_validation_dataset_names[dataloader_idx]
            zero_loss = torch.tensor(0.0, requires_grad=True, device=logits.device)
            self.log(f"offline_eval/{dataset_name}/val/loss", zero_loss, sync_dist=True)
            print("Validation Loss: 0.0 (no available views)")
            return zero_loss

        # Filter out samples with no available views
        valid_logits = logits[valid_samples]
        valid_labels = labels[valid_samples]
        valid_avail_mask = avail_mask[valid_samples]

        # Expand labels to match logits shape: (valid_batch_size, num_views, 1, num_labels)
        labels_expanded = valid_labels.unsqueeze(1).unsqueeze(2).expand_as(valid_logits)

        # Calculate loss for each view independently
        view_losses = self.loss_fn(
            valid_logits, labels_expanded
        )  # shape: (valid_batch_size, num_views, 1)

        # Apply availability mask and sum losses only for available views
        masked_losses = view_losses * valid_avail_mask.squeeze(
            -1
        )  # shape: (valid_batch_size, num_views, 1)
        loss = (
            masked_losses.sum() / valid_avail_mask.sum()
        )  # average across all available views

        # For metrics, simply average the logits across available views
        masked_logits = (
            valid_logits * valid_avail_mask
        )  # shape: (valid_batch_size, num_views, 1, num_labels)
        numerator = masked_logits.sum(dim=1)  # shape: (valid_batch_size, 1, num_labels)
        denominator = valid_avail_mask.sum(dim=1)  # shape: (valid_batch_size, 1, 1)

        # Average logits across available views (denominator > 0 guaranteed by filtering above)
        averaged_logits = numerator / denominator
        averaged_logits = averaged_logits.squeeze(
            1
        )  # shape: (valid_batch_size, num_labels)
        preds = torch.sigmoid(averaged_logits)

        dataset_name = self.oe_validation_dataset_names[dataloader_idx]

        if self.current_epoch % self.oe_auroc_frequency == 0:
            self.auroc_dict[dataset_name].update(preds, valid_labels.int())
            self.auroc_macro_dict[dataset_name].update(preds, valid_labels.int())
            self.accuracy_macro_dict[dataset_name].update(preds, valid_labels.int())
            self.f1_dict[dataset_name].update(preds, valid_labels.int())
            self.f1_macro_dict[dataset_name].update(preds, valid_labels.int())
            self.ap_macro_dict[dataset_name].update(preds, valid_labels.int())
            self.brier_macro_dict[dataset_name].update(preds, valid_labels.int())

        self.log(f"offline_eval/{dataset_name}/val/loss", loss, sync_dist=True)
        print("Validation Loss:", loss)
        return loss

    def on_validation_epoch_end(self):
        if self.current_epoch % self.oe_auroc_frequency == 0:
            for dataset_name in self.oe_validation_dataset_names:
                auroc_all = self.auroc_dict[dataset_name].compute()
                auroc_macro = self.auroc_macro_dict[dataset_name].compute()
                accuracy_macro_dict = self.accuracy_macro_dict[dataset_name].compute()
                f1_all = self.f1_dict[dataset_name].compute()
                f1_macro_dict = self.f1_macro_dict[dataset_name].compute()
                ap_macro_dict = self.ap_macro_dict[dataset_name].compute()
                brier_macro_dict = self.brier_macro_dict[dataset_name].compute()

                for i, v in enumerate(auroc_all):
                    self.log(
                        f"offline_eval/{dataset_name}/val/auroc_label_{i}",
                        v,
                        sync_dist=True,
                    )
                for i, v in enumerate(f1_all):
                    self.log(
                        f"offline_eval/{dataset_name}/val/f1_label_{i}",
                        v,
                        sync_dist=True,
                    )
                self.log(
                    f"offline_eval/{dataset_name}/val/auroc_macro",
                    auroc_macro,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"offline_eval/{dataset_name}/val/accuracy_macro",
                    accuracy_macro_dict,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"offline_eval/{dataset_name}/val/f1_macro",
                    f1_macro_dict,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"offline_eval/{dataset_name}/val/ap_macro",
                    ap_macro_dict,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"offline_eval/{dataset_name}/val/brier_macro",
                    brier_macro_dict,
                    prog_bar=True,
                    sync_dist=True,
                )

                self.auroc_dict[dataset_name].reset()
                self.auroc_macro_dict[dataset_name].reset()
                self.accuracy_macro_dict[dataset_name].reset()
                self.f1_dict[dataset_name].reset()
                self.f1_macro_dict[dataset_name].reset()
                self.ap_macro_dict[dataset_name].reset()
                self.brier_macro_dict[dataset_name].reset()
        else:
            print(
                "Skipping AUROC computation for this epoch, as it is not a multiple of oe_auroc_frequency"
            )

    def configure_optimizers(self):
        effective_bs = self.oe_batch_size * self.oe_num_gpus * self.oe_num_nodes
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.oe_base_learning_rate * effective_bs / 256,
            weight_decay=self.oe_learning_rate_weight_decay,
        )
        lr_scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.5,
            total_iters=self.oe_n_epochs_warmup,
        )
        lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.oe_n_epochs - self.oe_n_epochs_warmup,
            eta_min=0,
            last_epoch=-1,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[lr_scheduler1, lr_scheduler2],
            milestones=[self.oe_n_epochs_warmup],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "name": "offline_eval/lr",
            },
        }


class EnsembleClassifier(pl.LightningModule):
    def __init__(
        self,
        oe_batch_size,
        oe_num_nodes,
        oe_num_gpus,
        oe_n_epochs,
        oe_n_epochs_warmup,
        oe_learning_rate,
        oe_learning_rate_weight_decay,
        oe_encoders,
        oe_num_labels: int,
        oe_checkpoint_init: bool,
        oe_freeze_encoder: bool,
        oe_feature_policy: str,
        oe_ckpt_path: str,
        oe_clf_head_type: str,
        oe_logging_frequency: int,
        oe_auroc_frequency: int,
        oe_validation_dataset_names: list,
        oe_latent_attention_module,
        # max num views to consider per study
        oe_max_num_views: int,
    ):
        super().__init__()
        self.oe_max_num_views = oe_max_num_views
        self.oe_num_labels = oe_num_labels
        self.oe_checkpoint_init = oe_checkpoint_init
        self.oe_freeze_encoder = oe_freeze_encoder
        self.oe_feature_policy = oe_feature_policy
        self.oe_ckpt_path = oe_ckpt_path
        self.oe_clf_head_type = oe_clf_head_type
        self.oe_validation_dataset_names = oe_validation_dataset_names
        self.oe_logging_frequency = oe_logging_frequency
        self.oe_batch_size = oe_batch_size
        self.oe_num_nodes = oe_num_nodes
        self.oe_num_gpus = oe_num_gpus
        self.oe_n_epochs = oe_n_epochs
        self.oe_n_epochs_warmup = oe_n_epochs_warmup
        self.oe_base_learning_rate = oe_learning_rate
        self.oe_learning_rate_weight_decay = oe_learning_rate_weight_decay
        self.oe_auroc_frequency = oe_auroc_frequency

        # we save only the vision encoder, we don't use other modality encoders
        self.oe_vis_encoder = oe_encoders["vision"]
        self.save_hyperparameters(ignore=["oe_vis_encoder", "oe_encoders"])
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # CLF head definition
        emb_dim = self.oe_vis_encoder.emb_dim
        if oe_clf_head_type == "non_linear":
            self.clf_head = nn.Sequential(
                nn.Linear(emb_dim, emb_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(emb_dim // 2, oe_num_labels),
            )
        elif oe_clf_head_type == "linear":
            self.clf_head = nn.Linear(emb_dim, oe_num_labels)
        else:
            raise Exception("CLF HEAD TYPE NOT SUPPORTED")

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.auroc_dict = nn.ModuleDict(
            {
                name: MultilabelAUROC(num_labels=oe_num_labels, average=None)
                for name in self.oe_validation_dataset_names
            }
        )
        self.auroc_macro_dict = nn.ModuleDict(
            {
                name: MultilabelAUROC(num_labels=oe_num_labels, average="macro")
                for name in self.oe_validation_dataset_names
            }
        )
        self.accuracy_macro_dict = nn.ModuleDict(
            {
                name: MultilabelAccuracy(num_labels=oe_num_labels, average="macro")
                for name in self.oe_validation_dataset_names
            }
        )
        self.f1_dict = nn.ModuleDict(
            {
                name: MultilabelF1Score(num_labels=oe_num_labels, average=None)
                for name in self.oe_validation_dataset_names
            }
        )
        self.f1_macro_dict = nn.ModuleDict(
            {
                name: MultilabelF1Score(num_labels=oe_num_labels, average="macro")
                for name in self.oe_validation_dataset_names
            }
        )
        self.ap_macro_dict = nn.ModuleDict(
            {
                name: MultilabelAveragePrecision(
                    num_labels=oe_num_labels, average="macro"
                )
                for name in self.oe_validation_dataset_names
            }
        )
        self.brier_macro_dict = nn.ModuleDict(
            {
                name: MultilabelBrierScore(num_labels=oe_num_labels, average="macro")
                for name in self.oe_validation_dataset_names
            }
        )
        if self.oe_freeze_encoder:
            print("Freezing vision encoder weights")
            for param in self.oe_vis_encoder.parameters():
                param.requires_grad = False
        else:
            print("Full Fine-tuning of vision encoder")

    def select_features(self, feats):
        # adapted from: select_alignment_tokens(self, feats) MMVMMAE

        if self.oe_feature_policy == "cls":
            # use only CLS token
            lat = feats.movedim(0, 1)[:, 0, :].unsqueeze(1)
        else:
            raise ValueError("oe_feature_policy is implemented only as cls")
        return lat

    def forward(self, batch):
        data, labels = batch
        logits_list = []
        avail_mask_list = []
        enc_out = {}

        self.oe_vis_encoder.set_mask_ratio(0.0)

        for v_key, v_val in data["vision"].items():
            inputs = {}
            inputs["data"] = v_val["data"]
            inputs["modality_index"] = v_val["index"]
            e_m_out = self.oe_vis_encoder(inputs)
            enc_out[f"vision_{v_key}"] = e_m_out
            lat = self.select_features(e_m_out["features"])
            logits = self.clf_head(lat)
            logits_list.append(logits)
            avail_mask_list.append(v_val["mask"])

        logits = torch.stack(
            logits_list, dim=1
        )  # shape: (batch_size, num_views, 1, num_labels)
        avail_mask = (
            torch.stack(avail_mask_list, dim=1).unsqueeze(-1).unsqueeze(-1)
        )  # shape: (batch_size, num_views, 1, 1)

        # remove logits for views that are not available
        logits = logits * avail_mask  # shape: (batch_size, num_views, 1, num_labels)

        # aggregate logits - we do mean across available views
        numerator = logits.sum(dim=1)  # shape: (batch_size, 1, num_labels)
        denominator = avail_mask.sum(dim=1)

        # make sure there are no NaN in the denominator
        batch_mask = denominator.squeeze(-1).squeeze(-1) != 0
        numerator = numerator[batch_mask]
        denominator = denominator[batch_mask]
        labels = labels[batch_mask]

        logits_ensemble = (numerator / denominator).squeeze(1)

        return logits_ensemble, labels

    def training_step(self, batch, batch_idx):
        logits, labels = self.forward(batch)
        loss = self.loss_fn(logits, labels)
        print("Training loss:", loss)
        self.log("offline_eval/train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits, labels = self.forward(batch)
        loss = self.loss_fn(logits, labels)
        preds = torch.sigmoid(logits)
        dataset_name = self.oe_validation_dataset_names[dataloader_idx]

        if self.current_epoch % self.oe_auroc_frequency == 0:
            self.auroc_dict[dataset_name].update(preds, labels.int())
            self.auroc_macro_dict[dataset_name].update(preds, labels.int())
            self.accuracy_macro_dict[dataset_name].update(preds, labels.int())
            self.f1_dict[dataset_name].update(preds, labels.int())
            self.f1_macro_dict[dataset_name].update(preds, labels.int())
            self.ap_macro_dict[dataset_name].update(preds, labels.int())
            self.brier_macro_dict[dataset_name].update(preds, labels.int())

        self.log(f"offline_eval/{dataset_name}/val/loss", loss, sync_dist=True)
        print("Validation Loss:", loss)
        return loss

    def on_validation_epoch_end(self):
        if self.current_epoch % self.oe_auroc_frequency == 0:
            for dataset_name in self.oe_validation_dataset_names:
                auroc_all = self.auroc_dict[dataset_name].compute()
                auroc_macro = self.auroc_macro_dict[dataset_name].compute()
                accuracy_macro_dict = self.accuracy_macro_dict[dataset_name].compute()
                f1_all = self.f1_dict[dataset_name].compute()
                f1_macro_dict = self.f1_macro_dict[dataset_name].compute()
                ap_macro_dict = self.ap_macro_dict[dataset_name].compute()
                brier_macro_dict = self.brier_macro_dict[dataset_name].compute()

                for i, v in enumerate(auroc_all):
                    self.log(
                        f"offline_eval/{dataset_name}/val/auroc_label_{i}",
                        v,
                        sync_dist=True,
                    )
                for i, v in enumerate(f1_all):
                    self.log(
                        f"offline_eval/{dataset_name}/val/f1_label_{i}",
                        v,
                        sync_dist=True,
                    )
                self.log(
                    f"offline_eval/{dataset_name}/val/auroc_macro",
                    auroc_macro,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"offline_eval/{dataset_name}/val/accuracy_macro",
                    accuracy_macro_dict,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"offline_eval/{dataset_name}/val/f1_macro",
                    f1_macro_dict,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"offline_eval/{dataset_name}/val/ap_macro",
                    ap_macro_dict,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"offline_eval/{dataset_name}/val/brier_macro",
                    brier_macro_dict,
                    prog_bar=True,
                    sync_dist=True,
                )

                self.auroc_dict[dataset_name].reset()
                self.auroc_macro_dict[dataset_name].reset()
                self.accuracy_macro_dict[dataset_name].reset()
                self.f1_dict[dataset_name].reset()
                self.f1_macro_dict[dataset_name].reset()
                self.ap_macro_dict[dataset_name].reset()
                self.brier_macro_dict[dataset_name].reset()
        else:
            print(
                "Skipping AUROC computation for this epoch, as it is not a multiple of oe_auroc_frequency"
            )

    def configure_optimizers(self):
        effective_bs = self.oe_batch_size * self.oe_num_gpus * self.oe_num_nodes
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.oe_base_learning_rate * effective_bs / 256,
            weight_decay=self.oe_learning_rate_weight_decay,
        )
        lr_scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.5,
            total_iters=self.oe_n_epochs_warmup,
        )
        lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.oe_n_epochs - self.oe_n_epochs_warmup,
            eta_min=0,
            last_epoch=-1,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[lr_scheduler1, lr_scheduler2],
            milestones=[self.oe_n_epochs_warmup],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "name": "offline_eval/lr",
            },
        }


class StanfordViTUnimodalClassifier(UnimodalClassifier):
    """
    Classifier that swaps the vision encoder for Stanford CheXagent's ViT
    and otherwise behaves exactly like the base EnsembleClassifier.
    """

    def __init__(self, *args, oe_encoders=None, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        assert (
            cache_dir is not None
        ), "Please provide a cache_dir where CheXagent can be downloaded"
        # 1) Load CheXagent and grab its vision model
        dtype = torch.float16
        transformer_model = AutoModelForCausalLM.from_pretrained(
            "StanfordAIMI/CheXagent-8b",
            torch_dtype=dtype,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

        vision_transformer = transformer_model.vision_model

        stanford_adapter = StanfordVisionAdapter(vision_transformer)

        if oe_encoders is None:
            oe_encoders = {}
        else:
            oe_encoders = dict(oe_encoders)  # copy to avoid side-effects
        oe_encoders["vision"] = stanford_adapter

        # 4) Hand over to the normal EnsembleClassifier init
        super().__init__(oe_encoders=oe_encoders, *args, **kwargs)


class StanfordViTEnsembleClassifier(EnsembleClassifier):
    """
    Classifier that swaps the vision encoder for Stanford CheXagent's ViT
    and otherwise behaves exactly like the base EnsembleClassifier.
    """

    def __init__(self, *args, oe_encoders=None, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        assert (
            cache_dir is not None
        ), "Please provide a cache_dir where CheXagent can be downloaded"
        # 1) Load CheXagent and grab its vision model
        dtype = torch.float16
        transformer_model = AutoModelForCausalLM.from_pretrained(
            "StanfordAIMI/CheXagent-8b",
            torch_dtype=dtype,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

        vision_transformer = transformer_model.vision_model

        stanford_adapter = StanfordVisionAdapter(vision_transformer)

        if oe_encoders is None:
            oe_encoders = {}
        else:
            oe_encoders = dict(oe_encoders)  # copy to avoid side-effects
        oe_encoders["vision"] = stanford_adapter

        # 4) Hand over to the normal EnsembleClassifier init
        super().__init__(oe_encoders=oe_encoders, *args, **kwargs)


class StanfordVisionAdapter(nn.Module):
    """
    Wrapper to make CheXagent's vision transformer compatible with our vision encoder:
      - has .emb_dim
      - has set_mask_ratio()
      - forward(inputs) -> {"features": [tokens, batch, dim]}
    """

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model
        self.vision_config = vision_model.config
        """
            1: CheXagentVisionConfig {
            1:   "_attn_implementation_autoset": true,
            1:   "attention_dropout": 0.0,
            1:   "hidden_act": "gelu",
            1:   "hidden_size": 1408,
            1:   "image_size": 448,
            1:   "initializer_range": 1e-10,
            1:   "intermediate_size": 6144,
            1:   "layer_norm_eps": 1e-06,
            1:   "model_type": "chexagent_vision_model",
            1:   "num_attention_heads": 16,
            1:   "num_hidden_layers": 40,
            1:   "patch_size": 14,
            1:   "qkv_bias": true,
            1:   "transformers_version": "4.51.3"
            1: }
        """
        self.emb_dim = self.vision_config.hidden_size
        print("StanfordVisionAdapter EMB_DIM: ", str(self.emb_dim))

    def set_mask_ratio(self, x):
        # kept for compatibility
        pass

    def forward(self, inputs: dict):

        x = inputs["data"]

        # make it 3 channels
        assert x.shape[1] == 3, "Stanford ViT expects 3-channel images"
        assert (
            x.shape[2] == 448 and x.shape[3] == 448
        ), "Stanford ViT expects 448x448 images"

        with autocast(dtype=torch.float16):
            out = self.vision_model(pixel_values=x, output_hidden_states=False)

        feats = out[0]  # [B, T, D]
        feats = feats.permute(
            1, 0, 2
        ).contiguous()  # -> [T, B, D] as select_features expects
        return {"features": feats}


class BiomedClipEnsembleClassifier(EnsembleClassifier):
    """
    Classifier that swaps the vision encoder for BiomedCLIP's ViT
    and otherwise behaves exactly like the base EnsembleClassifier.
    """

    def __init__(
        self,
        *args,
        oe_encoders=None,
        vision_model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        **kwargs,
    ):
        cache_dir = kwargs.pop("cache_dir", None)
        assert (
            cache_dir is not None
        ), "Please provide a cache_dir where CheXagent can be downloaded"
        # Load BiomedCLIP model
        model, _ = create_model_from_pretrained(vision_model_name, cache_dir)
        vision_encoder = model.visual

        biomed_adapter = BiomedClipVisionAdapter(vision_encoder)

        if oe_encoders is None:
            oe_encoders = {}
        else:
            oe_encoders = dict(oe_encoders)
        oe_encoders["vision"] = biomed_adapter
        super().__init__(oe_encoders=oe_encoders, *args, **kwargs)


class BiomedClipUnimodalClassifier(UnimodalClassifier):
    """
    Classifier that swaps the vision encoder for BiomedCLIP's ViT
    and otherwise behaves exactly like the base EnsembleClassifier.
    """

    def __init__(
        self,
        *args,
        oe_encoders=None,
        vision_model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        **kwargs,
    ):
        cache_dir = kwargs.pop("cache_dir", None)
        assert (
            cache_dir is not None
        ), "Please provide a cache_dir where CheXagent can be downloaded"
        # Load BiomedCLIP model
        model, _ = create_model_from_pretrained(vision_model_name, cache_dir)
        vision_encoder = model.visual

        biomed_adapter = BiomedClipVisionAdapter(vision_encoder)

        if oe_encoders is None:
            oe_encoders = {}
        else:
            oe_encoders = dict(oe_encoders)
        oe_encoders["vision"] = biomed_adapter
        super().__init__(oe_encoders=oe_encoders, *args, **kwargs)


class BiomedClipVisionAdapter(nn.Module):
    """
    Wrapper to make BiomedCLIP's vision transformer compatible with our vision encoder:
      - has .emb_dim
      - has set_mask_ratio()
      - forward(inputs) -> {"features": [tokens, batch, dim]}
    """

    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.emb_dim = 512  # projection dimension
        print("BiomedClipVisionAdapter EMB_DIM: ", str(self.emb_dim))

    def set_mask_ratio(self, x):
        # kept for compatibility
        pass

    def forward(self, inputs: dict):
        x = inputs["data"]
        assert x.shape[1] == 3, "BiomedCLIP expects 3-channel images"
        out = self.vision_encoder(x)
        feats = out.unsqueeze(0)  # [1, B, D]
        return {"features": feats}
