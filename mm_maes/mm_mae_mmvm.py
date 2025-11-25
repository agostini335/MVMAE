import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mm_maes.mm_mae import MMMAE
from mm_maes.utils import ContrastiveLoss
from einops import rearrange
from utils.generic import (
    get_text_modalities,
    has_text_modality,
    is_text_modality,
    remove_text_modalities_from_list,
)
from hydra.utils import instantiate
from einops import repeat, rearrange


class MMVMMAE(MMMAE):
    def __init__(
        self,
        encoders,
        decoders,
        evaluator,
        annealingmodule,
        vision_to_text_decoder,
        vision_to_text_decoder_module,
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
        offline_eval,
        patch_size,
        mask_ratio,
        regularization_tokens,
        use_latent_heads,
        regularization_metric,
        pairwise_regularization,
        compute_rec,
        temperature,
        sync_logs_on_epoch_end,
        use_bce_loss,
        log_debug,
        validation_dataset_names=None,
    ):
        super().__init__(
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
            offline_eval,
            sync_logs_on_epoch_end,
            validation_dataset_names,
        )

        enc_keys = list(encoders.keys())
        if has_text_modality(enc_keys):
            tm = get_text_modalities(text_modality_config, vision_modality_config)[0]
            self.emb_dim_text = encoders[tm].model.config.hidden_size
            remove_text_modalities_from_list(enc_keys)
            assert self.emb_dim_text == encoders[tm].emb_dim
        assert not has_text_modality(enc_keys)

        self.emb_dim = encoders["vision"].emb_dim
        self.img_size = encoders["vision"].image_size

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.use_latent_heads = use_latent_heads
        self.regularization_tokens = regularization_tokens
        self.regularization_metric = regularization_metric
        self.pairwise_regularization = pairwise_regularization
        self.compute_rec = compute_rec
        self.vision_to_text_decoder = vision_to_text_decoder
        self.temperature = temperature
        self.simclr_loss = ContrastiveLoss(self.temperature)
        self.log_debug = log_debug

        if self.use_latent_heads:
            self.latent_heads = nn.ModuleDict()
            if vision_modality_config is not None:
                self.latent_heads.add_module(
                    "vision", nn.Linear(self.emb_dim, self.emb_dim)
                )
            if text_modality_config is not None:
                self.latent_heads.add_module(
                    "text", nn.Linear(self.emb_dim_text, self.emb_dim)
                )
        if self.vision_to_text_decoder:
            self.vision_to_text_reconstruction_text = vision_to_text_decoder.text_field
            self.text_decoder = vision_to_text_decoder_module
            self.gamma = self.vision_to_text_decoder.loss_weighting_factor

        if self.regularization_metric == "mix":
            self.a = torch.tensor(0.33, requires_grad=False)
            self.b = torch.tensor(0.33, requires_grad=False)
            self.c = torch.tensor(0.33, requires_grad=False)

        self.beta_annealing = annealingmodule
        self.save_hyperparameters(
            ignore=["encoders", "decoders", "vision_to_text_decoder_module"]
        )

    def forward(self, batch):
        data = batch[0]
        # encode modalities
        enc_out = {}
        latent_heads_out = {}
        # decode modalities

        # vision block
        vision_recs = {}
        vision_masks = {}
        if self.vision_modality_config is not None:
            vis_encoder = self.encoders["vision"]
            vis_encoder.set_mask_ratio(self.mask_ratio)
            vis_decoder = self.decoders["vision"]
            if self.vision_to_text_decoder is not None:
                text_dec_inputs = {}
                text_dec_inputs["labels"] = data["text"][
                    self.vision_to_text_reconstruction_text
                ]["data"]["labels"]
                v_text_avail = data["text"][self.vision_to_text_reconstruction_text][
                    "mask"
                ]
                if not getattr(self.text_decoder, "cappa_per_view", True):
                    text_dec_inputs["backward_indexes"] = []
                    text_dec_inputs["latent_tokens"] = []
            for v_key, v_val in data["vision"].items():
                inputs = {}
                inputs["data"] = v_val["data"]
                inputs["modality_index"] = v_val["index"]
                e_m_out = vis_encoder(inputs)  # mod_m for text must be a dict
                enc_out[f"vision_{v_key}"] = e_m_out

                if self.vision_modality_config.decode is not None:
                    if "self" in self.vision_modality_config.decode:
                        inputs = e_m_out
                        inputs["modality_index"] = v_val["index"]
                        dec_m_out = vis_decoder(inputs)

                        if self.hparams.use_bce_loss:
                            dec_m_out["img"] = torch.sigmoid(dec_m_out["img"])
                        vision_recs[f"vision_{v_key}"] = dec_m_out["img"]
                        vision_masks[f"vision_{v_key}"] = dec_m_out["mask"]
                    if self.vision_to_text_decoder is not None:
                        modality_embeddings = torch.zeros_like(e_m_out["features"])
                        if vis_decoder.use_modality_emb:
                            modality_index = inputs["modality_index"]
                            modality_embeddings = vis_decoder.mod_embedding[
                                modality_index
                            ]
                        if not getattr(self.text_decoder, "cappa_per_view", True):
                            text_dec_inputs["backward_indexes"].append(
                                inputs["backward_indexes"]
                            )
                            text_dec_inputs["latent_tokens"].append(
                                e_m_out["features"] + modality_embeddings
                            )
                        else:
                            text_dec_inputs["backward_indexes"] = inputs[
                                "backward_indexes"
                            ]
                            text_dec_inputs["latent_tokens"] = (
                                e_m_out["features"] + modality_embeddings
                            )
                            dec_m_text_out, preds, gts = self.text_decoder(
                                text_dec_inputs
                            )
                            dec_m_text_out_masked = torch.mul(
                                v_text_avail.unsqueeze(1), dec_m_text_out
                            )
                            vision_recs[f"vision_{v_key}_to_text"] = (
                                dec_m_text_out_masked
                            )
                            vision_recs[f"vision_{v_key}_to_text_preds"] = preds
                            vision_recs[f"vision_{v_key}_to_text_gts"] = gts
                            vision_recs[f"vision_{v_key}_to_text_inputs"] = {
                                k: v.detach() for k, v in text_dec_inputs.items()
                            }
                        

                lat_v = self.select_alignment_tokens(e_m_out["features"])
                lat_out_v = self.prepare_features_for_alignment("vision", lat_v)
                latent_heads_out[f"vision_{v_key}"] = lat_out_v
            if self.vision_to_text_decoder is not None and not getattr(
                self.text_decoder, "cappa_per_view", True
            ):
                dec_m_text_out, preds, gts = self.text_decoder(text_dec_inputs)
                dec_m_text_out_masked = torch.mul(
                    v_text_avail.unsqueeze(1), dec_m_text_out
                )
                vision_recs["vision_to_text"] = dec_m_text_out_masked
                vision_recs["vision_to_text_preds"] = preds
                vision_recs["vision_to_text_gts"] = gts
                vision_recs["vision_to_text_inputs"] = {
                    k: (
                        v.detach()
                        if not isinstance(v, list)
                        else [x.detach() for x in v]
                    )
                    for k, v in text_dec_inputs.items()
                }

        # text block
        text_losses = {}
        if self.text_modality_config is not None:
            text_model = self.encoders["text"]
            for t_key, t_val in data["text"].items():
                inputs = {}
                inputs["data"] = t_val["data"]
                inputs["modality_index"] = t_val["index"]
                e_m_out = text_model(inputs)  # mod_m for text must be a dict
                enc_out[f"text_{t_key}"] = e_m_out["features"]
                lat_t = self.select_alignment_tokens(
                    rearrange(e_m_out["features"], "b t e-> t b e")
                )
                lat_out_t = self.prepare_features_for_alignment("text", lat_t)
                latent_heads_out[f"text_{t_key}"] = lat_out_t
                if self.text_modality_config.decode is not None:
                    if "self" in self.text_modality_config.decode and self.compute_rec:
                        text_losses[f"text_{t_key}"] = e_m_out["loss"]

        return vision_recs, vision_masks, enc_out, latent_heads_out, text_losses

    def prepare_features_for_alignment(self, modality_key, feats):
        if not self.pairwise_regularization:
            feats = torch.mean(feats, dim=1)

        if self.use_latent_heads:
            feats = self.latent_heads[modality_key](feats)
        return feats

    def select_alignment_tokens(self, feats):
        # feats need to have shape (n_tokens, batch_size, emb_dim)
        if self.regularization_tokens == "cls":
            # use only CLS token
            lat = feats.movedim(0, 1)[:, 0, :].unsqueeze(1)
        elif self.regularization_tokens == "all_but_cls":
            lat = feats.movedim(0, 1)[:, 1:, :]
        elif self.regularization_tokens == "all":
            # use all tokens
            lat = feats.movedim(0, 1)
        else:
            raise ValueError("regularization tokens must be in [cls, all, all_but_cls]")
        return lat

    def compute_loss(self, str_set, data, preds, masks, latent_heads_out, text_losses):
        mod_masks = {}
        log_dict = {}
        losses = []
        total_loss = torch.zeros((), device=self.device)
        # vision
        if self.vision_modality_config is not None:
            losses_vtt = []
            vision_losses = []
            if self.vision_modality_config.decode is not None:
                for v_key, v_val in data["vision"].items():
                    if "self" in self.vision_modality_config.decode:
                        v_data = v_val["data"]
                        v_mask_available = v_val["mask"]
                        mod_masks[f"vision_{v_key}"] = v_mask_available
                        v_pred = preds[f"vision_{v_key}"]
                        v_mask_enc = masks[f"vision_{v_key}"]
                        if self.hparams.use_bce_loss:
                            if torch.any(v_data > 1) or torch.any(v_data < 0):
                                raise ValueError(
                                    "use_bse_loss currently not supported whit data normalization enabled."
                                )
                            loss_v_samples = F.binary_cross_entropy(
                                v_pred, v_data, reduction="none"
                            )
                            loss_v_samples = (loss_v_samples * v_mask_enc).mean(
                                dim=[1, 2, 3]
                            )
                        else:
                            loss_v_samples = torch.mean(
                                ((v_pred - v_data) ** 2 * v_mask_enc) / self.mask_ratio,
                                dim=[1, 2, 3],
                            )
                        loss_v = torch.mean(torch.mul(v_mask_available, loss_v_samples))
                        vision_losses.append(loss_v.unsqueeze(dim=0))
                        losses.append(loss_v.unsqueeze(dim=0))

                    if self.vision_to_text_decoder:
                        if f"vision_{v_key}_to_text" in preds:
                            loss_v_text_samples = torch.mean(
                                preds[f"vision_{v_key}_to_text"],
                                dim=[1],
                            )
                        else:
                            loss_v_text_samples = torch.mean(
                                preds["vision_to_text"],
                                dim=[1],
                            )

                        loss_v_text = torch.mean(
                            torch.mul(v_mask_available, loss_v_text_samples)
                        )
                        losses_vtt.append(loss_v_text.unsqueeze(0))
                        losses.append(self.gamma * loss_v_text.unsqueeze(0))

            # log vision loss
            total_vision_loss = torch.cat(vision_losses).mean()
            log_dict[str_set + "/loss/rec_loss_vision"] = total_vision_loss
            # log vision-to-text loss
            if self.vision_to_text_decoder:
                total_vtt_loss = torch.cat(losses_vtt).mean()
                log_dict[str_set + "/loss/rec_loss_visiontotext"] = total_vtt_loss

        if self.text_modality_config is not None:
            if self.text_modality_config.decode is not None:
                if "self" in self.text_modality_config.decode:
                    text_rec_losses = []
                    for t_key, t_val in data["text"].items():
                        t_mask_available = t_val["mask"]
                        mod_masks[f"text_{t_key}"] = t_mask_available
                        if self.compute_rec:
                            # handle text loss
                            log_dict[f"{str_set}/loss/rec_loss_{t_key}"] = text_losses[
                                f"text_{t_key}"
                            ]

                            losses.append(text_losses[f"text_{t_key}"].unsqueeze(dim=0))
                            text_rec_losses.append(
                                text_losses[f"text_{t_key}"].unsqueeze(dim=0)
                            )
                    if self.compute_rec:
                        total_text_loss = torch.cat(text_rec_losses).mean()
                        log_dict[str_set + "/loss/rec_loss_text"] = total_text_loss

        total_loss = torch.cat(losses).mean()
        log_dict[str_set + "/loss/rec_loss"] = total_loss

        cls_mse = self.feats_mse(str_set, latent_heads_out, mod_masks, log_dict)
        cls_cosine = self.feats_cos_sim(str_set, latent_heads_out, mod_masks, log_dict)
        if not self.pairwise_regularization:
            cls_simclr = self.feats_simclr(
                str_set, latent_heads_out, mod_masks, log_dict
            )
        if self.regularization_metric == "mse":
            loss_feats_avg = cls_mse
        elif self.regularization_metric == "cosine_sim":
            loss_feats_avg = cls_cosine
        elif self.regularization_metric in ("simclr"):
            loss_feats_avg = cls_simclr
        elif self.regularization_metric == "mix":
            loss_feats_avg = (
                self.a * cls_mse + self.b * cls_cosine + self.c * cls_simclr
            )
            log_dict[str_set + "/loss/w_a"] = self.a
            log_dict[str_set + "/loss/w_b"] = self.b
            log_dict[str_set + "/loss/w_c"] = self.c
            log_dict[str_set + "/loss/mix"] = loss_feats_avg
        else:
            loss_feats_avg = torch.zeros((), device=self.device)

        beta_tmp = self.beta_annealing.get_beta_value(
            self.current_epoch,
        )
        total_loss += beta_tmp * loss_feats_avg
        log_dict["beta"] = beta_tmp
        log_dict[str_set + "/loss/loss"] = total_loss

        if not self.training:
            self.log_dict(
                log_dict,
                sync_dist=True,
            )
        elif self.trainer.is_global_zero:
            self.log_dict(
                log_dict,
                sync_dist=True,
                on_step=not self.hparams.sync_logs_on_epoch_end,
                on_epoch=self.hparams.sync_logs_on_epoch_end,
                rank_zero_only=True,
            )
        return total_loss

    def feats_mse(self, str_set, feats, mod_masks, log_dict):
        n_mods = len(feats)
        t_mod_masks = torch.cat([m_val.unsqueeze(0) for m_val in mod_masks.values()])
        t_feats = torch.cat([f_val.unsqueeze(0) for f_val in feats.values()])
        t_feats = nn.functional.normalize(t_feats, dim=-1)
        t_feats1, t_feats2 = t_feats.unsqueeze(0), t_feats.unsqueeze(1)
        t_mod_masks1, t_mod_masks2 = t_mod_masks.unsqueeze(0), t_mod_masks.unsqueeze(1)
        t_mod_masks_all = torch.logical_and(t_mod_masks1, t_mod_masks2)
        t_feats_diff = ((t_feats1 - t_feats2) ** 2).mean(dim=-1)
        if t_feats_diff.ndim > 3:
            t_mod_masks_all = t_mod_masks_all.unsqueeze(3)
        t_diff_all = t_mod_masks_all * t_feats_diff
        t_diff = t_diff_all.sum(dim=[0, 1]) / (n_mods**2)
        mse = t_diff.mean()
        log_dict[str_set + "/loss/cls_mse"] = mse
        return mse

    def feats_cos_sim(self, str_set, feats, mod_masks, log_dict):
        n_mods = len(feats)
        t_mod_masks = torch.cat([m_val.unsqueeze(0) for m_val in mod_masks.values()])
        t_mod_masks1, t_mod_masks2 = t_mod_masks.unsqueeze(0), t_mod_masks.unsqueeze(1)
        t_mod_masks_all = torch.logical_and(t_mod_masks1, t_mod_masks2)
        t_feats = torch.cat([f_val.unsqueeze(0) for f_val in feats.values()])
        t_feats = nn.functional.normalize(t_feats, dim=-1)
        t_feats1, t_feats2 = t_feats.unsqueeze(0), t_feats.unsqueeze(1)
        t_cos_sim = (t_feats1 * t_feats2).sum(dim=-1)
        if t_cos_sim.ndim > 3:
            t_mod_masks_all = t_mod_masks_all.unsqueeze(3)
        t_cos_all = t_mod_masks_all * t_cos_sim
        t_nondiags = torch.ones(
            (n_mods, n_mods), device=self.device
        ) - torch.diag_embed(torch.ones(n_mods, device=self.device), dim1=0, dim2=1)
        if t_cos_all.ndim > 3:
            t_nondiags = t_nondiags.unsqueeze(-1)
        t_cos = (t_nondiags.unsqueeze(-1) * t_cos_all).sum(dim=[0, 1]) / (
            n_mods**2 - n_mods
        )
        neg_cos_sim = -t_cos.mean()
        log_dict[str_set + "/loss/cls_cos_sim"] = neg_cos_sim
        return neg_cos_sim

    def feats_simclr(self, str_set, feats, mod_masks, log_dict):
        simclr_loss = []
        for (
            key_m,
            feat_m,
        ) in (
            feats.items()
        ):
            mod_mask_m = mod_masks[key_m]
            for key_mtilde, feat_mtilde in feats.items():
                if (
                    key_m == key_mtilde
                ):  # Skip computing loss if modalities are identical
                    continue
                mod_mask_mtilde = mod_masks[key_mtilde]
                loss_m_mtilde = self.simclr_loss(
                    feat_m.squeeze(1),
                    feat_mtilde.squeeze(1),
                    mod_mask_m,
                    mod_mask_mtilde,
                    self.device,
                )
                simclr_loss.append(loss_m_mtilde.unsqueeze(0))
        simclr_loss = torch.cat(simclr_loss, dim=0).mean()
        log_dict[str_set + "/loss/cls_simclr"] = simclr_loss
        return simclr_loss
