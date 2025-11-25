import pytorch_lightning as pl
import torch
import torch.distributed as dist
from einops import rearrange
from torch.distributed.nn.functional import all_gather

import wandb
from utils.generic import move_tensors_to_cpu, move_tensors_to_device


class MMMAE(pl.LightningModule):
    def __init__(
        self,
        encoders,
        decoders,
        evaluator,
        vision_modality_config,
        text_modality_config,
        learning_rate,
        learning_rate_weight_decay,
        batch_size,
        num_nodes,
        num_gpus,
        num_samples_lr_train,
        n_epochs,
        n_epochs_warmup,
        sync_logs_on_epoch_end,
        offline_eval=False,
        validation_dataset_names=None,
    ):
        super().__init__()
        self.encoders = encoders
        self.decoders = decoders
        self.evaluator = evaluator
        self.vision_modality_config = vision_modality_config
        self.text_modality_config = text_modality_config
        self.offline_eval = offline_eval
        self.num_samples_lr_train = num_samples_lr_train
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.num_gpus = num_gpus
        self.n_epochs = n_epochs
        self.n_epochs_warmup = n_epochs_warmup
        self.base_learning_rate = learning_rate
        self.learning_rate_weight_decay = learning_rate_weight_decay
        self.training_encodings = []
        self.validation_step_outputs = {}
        self.validation_dataset_names = validation_dataset_names
        self.rank_zero_train_dl = None  # use only from rank zero process
        self.first_val_batch_dict = {}
        self.save_hyperparameters(
            ignore=["encoders", "decoders", "vision_to_text_decoder_module"]
        )

    def _gather_tensor(self, tensor, concat_dim=0):
        """
        Gathers a tensor from all processes while preserving the autograd graph.
        """
        if dist.is_available() and dist.is_initialized():
            tensor = all_gather(tensor)
            return torch.cat(tensor, dim=concat_dim)
        return tensor

    def assign_rank_zero_train_dl(self, train_dl):
        self.rank_zero_train_dl = train_dl
        assert self.rank_zero_train_dl is not None
        print("assigned rank zero train data loader")

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        mods, labels = batch  # mods, labels, masks = batch
        mods_rec, masks, enc_out, latent_heads_out, text_losses = out
        loss = self.compute_loss(
            "train", mods, mods_rec, masks, latent_heads_out, text_losses
        )
        if self.compute_rec is False:
            latent_heads_out = {
                k: v.detach() for k, v in latent_heads_out.items()
            }
        return loss

    """
    def get_encodings(self, batch, device=None):
        data = batch[0]
        enc_out = {}
        if self.vision_modality_config is not None:
            vis_encoder = self.encoders["vision"]
            vis_encoder.set_mask_ratio(0.0)
            vis_encs = []
            for v_key, v_val in data["vision"].items():
                inputs = {}
                inputs["data"] = v_val["data"]
                inputs["modality_index"] = v_val["index"]
                if device is not None:
                    inputs = move_tensors_to_device(inputs, device)
                e_v_out = vis_encoder(inputs)  
                vis_encs.append(e_v_out["features"].unsqueeze(1))
            enc_out[f"vision"] = torch.cat(vis_encs, dim=1).mean(dim=1)
        if self.text_modality_config is not None:
            text_model = self.encoders["text"]
            text_encs = []
            for t_key, t_val in data["text"].items():
                inputs = {}
                inputs["data"] = t_val["data"]
                inputs["modality_index"] = t_val["index"]
                if device is not None:
                    inputs = move_tensors_to_device(inputs, device)
                e_t_out = text_model(inputs)  # mod_m for text must be a dict
                text_encs.append(rearrange(e_t_out["features"], "b t e-> t b e").unsqueeze(1))
            enc_out[f"text"] = torch.cat(text_encs, dim=1).mean(dim=1)
        return enc_out
    """

    def get_encodings(self, batch, device=None):
        data = batch[0]
        enc_out = {}
        if self.vision_modality_config is not None:
            vis_encoder = self.encoders["vision"]
            vis_encoder.set_mask_ratio(0.0)
            vis_encs = []
            v_masks = []
            for v_key, v_val in data["vision"].items():
                inputs = {}
                inputs["data"] = v_val["data"]
                inputs["modality_index"] = v_val["index"]
                if device is not None:
                    inputs = move_tensors_to_device(inputs, device)
                    v_val_mask = move_tensors_to_device(v_val["mask"], device)
                else:
                    v_val_mask = v_val["mask"]
                v_masks.append(v_val_mask.unsqueeze(1))
                e_v_out = vis_encoder(inputs)
                vis_encs.append(e_v_out["features"].unsqueeze(1))
            v_masks = rearrange(torch.cat(v_masks, dim=1), "b m -> 1 m b 1")
            v_embs = torch.cat(vis_encs, dim=1)
            n_avail_s = v_masks.sum(dim=1)
            v_embs_f = (v_embs * v_masks).sum(dim=1) / n_avail_s
            enc_out["vision"] = v_embs_f
        if self.text_modality_config is not None:
            text_model = self.encoders["text"]
            text_encs = []
            for t_key, t_val in data["text"].items():
                inputs = {}
                inputs["data"] = t_val["data"]
                inputs["modality_index"] = t_val["index"]
                if device is not None:
                    inputs = move_tensors_to_device(inputs, device)
                e_t_out = text_model(inputs)  # mod_m for text must be a dict
                text_encs.append(
                    rearrange(e_t_out["features"], "b t e-> t b e").unsqueeze(1)
                )
            enc_out["text"] = torch.cat(text_encs, dim=1).mean(dim=1)
        return enc_out

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = self.forward(batch)
        mods, labels = batch  # mods, labels, masks = batch
        mods_rec, masks, _, latent_heads_out, text_losses = out
        loss = self.compute_loss(
            "val", mods, mods_rec, masks, latent_heads_out, text_losses
        )

        # computing and gathering the encodings only if we are in the logging_frequency_lr epoch
        if (self.current_epoch + 1) % self.evaluator.logging_frequency_lr == 0:
            enc_emb = self.get_encodings(batch)
            for key in enc_emb.keys():
                if self.trainer.world_size > 1:
                    enc_emb[key] = self._gather_tensor(enc_emb[key], concat_dim=1)
                enc_emb[key] = move_tensors_to_cpu(enc_emb[key])
            if self.trainer.world_size > 1:
                labels = self._gather_tensor(labels)

            # get dataset name from idx
            d_name = self.validation_dataset_names[dataloader_idx]
            self.validation_step_outputs[d_name].append(
                [enc_emb, move_tensors_to_cpu(labels)]
            )
            if (
                self.trainer.is_global_zero
                and self.first_val_batch_dict[d_name] is None
            ):
                self.first_val_batch_dict[d_name] = batch
                print(f"First validation batch for dataset {d_name} saved")
        return loss

    def on_validation_epoch_start(self):
        # assertion and cleaning before validation epoch starts
        if not self.trainer.is_global_zero:
            self.rank_zero_train_dl = None
        else:
            assert self.rank_zero_train_dl is not None
        # clean up previous validation step outputs
        self.validation_step_outputs = {
            name: [] for name in self.validation_dataset_names
        }
        self.first_val_batch_dict = {
            name: None for name in self.validation_dataset_names
        }

    def on_validation_epoch_end(self):
        # skipping in sanity checking
        if self.trainer.sanity_checking:
            return

        # if we are not in the logging_frequency_lr epoch, we skip the online evaluation
        if (self.current_epoch + 1) % self.evaluator.logging_frequency_lr != 0:
            return

        log_dict = {}

        # Online Evaluation Phase - We work only on the rank zero process
        if self.trainer.is_global_zero:
            # assert that we have the rank zero train data loader and training encodings have been cleared
            assert self.rank_zero_train_dl is not None
            assert len(self.training_encodings) == 0

            print("Starting online evaluation phase on RANK ZERO process")
            was_training = self.training
            self.eval()

            with torch.no_grad():
                n_encodings = 0
                get_train_encodings_iter = 0
                for batch in self.rank_zero_train_dl:
                    mods, labels = batch
                    enc_out = self.get_encodings(batch, self.device)
                    for key in enc_out.keys():
                        # enc_out[key] = {
                        #     k: v.detach()
                        #     for k, v in enc_out[key].items()
                        #     if k in ["features", "cls_embedding"]
                        # }
                        enc_out[key] = move_tensors_to_cpu(enc_out[key])

                    n_encodings += labels.size(0)
                    self.training_encodings.append(
                        (enc_out, move_tensors_to_cpu(labels))
                    )
                    if n_encodings >= self.num_samples_lr_train:
                        break
                    get_train_encodings_iter += 1

            print("Total number of training encodings gathered:", n_encodings)

            # eval and logging
            logs = self.evaluator.eval(self)
            if "scalars" in logs:
                for key, value in logs["scalars"].items():
                    log_dict[key] = value
            if "imgs" in logs:
                for key, img in logs["imgs"].items():
                    if isinstance(img, tuple):
                        n_imgs = img[0].shape[0]
                        imgs = [
                            wandb.Image(img[0][i], caption=img[1][i])
                            for i in range(n_imgs)
                        ]
                    else:
                        imgs = [wandb.Image(img)]
                    self.logger.log_image(
                        key=key,
                        images=imgs
                    )
            # restorting the training state
            if was_training:
                self.train()


        # Broadcast the log_dict computed on rank zero to all other processes
        # so every process has the same log entries before logging. Use
        # torch.distributed.broadcast_object_list which can broadcast
        # picklable Python objects. Keep a safe fallback in case of errors.
        if dist.is_available() and dist.is_initialized():
            try:
                obj = [log_dict]
                dist.broadcast_object_list(obj, src=0)
                log_dict = obj[0]
            except Exception:
                # If broadcast fails for any reason, continue with the
                # local log_dict (likely empty on non-zero ranks).
                pass

        self.log_dict(
            log_dict,
            sync_dist=True,
            rank_zero_only=True,
        )

        self.training_encodings.clear()

    def configure_optimizers(self):
        effective_bs = self.batch_size * self.num_gpus * self.num_nodes
        eff_lr = self.base_learning_rate * effective_bs / 256 
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=eff_lr,
            weight_decay=self.learning_rate_weight_decay,
        )
        lr_scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.5,
            total_iters=self.n_epochs_warmup,
        )
        lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.n_epochs - self.n_epochs_warmup,
            eta_min=0*eff_lr,
            last_epoch=-1,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[lr_scheduler1, lr_scheduler2],
            milestones=[self.n_epochs_warmup],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
