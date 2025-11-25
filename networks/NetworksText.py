import random
from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Block
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoTokenizer, BertConfig, BertForMaskedLM

from config.networks.NetworkConfig import TokenizerModuleConfig
from networks.NetworksMAE import take_indexes


class BiomedBert_Encoder(pl.LightningModule):
    """
    Bert style encoder - it accepts BertForMaskedLM pretrained models.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        random_init: bool,
        hidden_state_index: int,
        freeze_all: bool,
        emb_dim: int,
        compute_rec: bool = True,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.pretrained_model_name = pretrained_model_name
        self.random_init = random_init

        if self.random_init:
            config = AutoConfig.from_pretrained(self.pretrained_model_name)
            self.model = BertForMaskedLM(config)
        else:
            self.model = BertForMaskedLM.from_pretrained(
                self.pretrained_model_name, use_safetensors=True
            )

        print(self.model)

        self.compute_rec = compute_rec
        self.hidden_state_index = hidden_state_index
        self.freeze_all = freeze_all

        if self.freeze_all:
            for param in self.model.bert.parameters():
                param.requires_grad = False

            for param in self.model.cls.parameters():
                param.requires_grad = False

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("Trainable:", name)

    def forward(self, inputs):
        inputs_data = inputs["data"]
        model_inputs = {
            "input_ids": inputs_data["input_ids"],
            "attention_mask": inputs_data["attention_mask"],
            "output_hidden_states": True,
        }
        if self.compute_rec:
            model_inputs["labels"] = inputs_data["labels"]
        else:
            model_inputs["labels"] = None
        outputs = self.model(**model_inputs)

        loss = outputs["loss"] if self.compute_rec else None

        feats_embedding = outputs["hidden_states"][self.hidden_state_index]
        return {"loss": loss, "features": feats_embedding}

    def set_mask_ratio(self, ratio):
        # DUMMY MASKED RATIO SETUP
        pass


class BiomedBert_Decoder(pl.LightningModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        print("TEXT DECODER INIT")

    def forward(self, inputs):
        return None


class SimpleVisionToTextDecoder(pl.LightningModule):
    """
    Bert Decoder.
    """

    def __init__(
        self,
        emb_dim: int,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        config = BertConfig()
        config.hidden_size = self.emb_dim
        self.config = config
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pad_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        # transformation
        self.tf_act = nn.GELU()
        self.tf_dense = nn.Linear(emb_dim, emb_dim)
        self.tf_ln = nn.LayerNorm(emb_dim, eps=config.layer_norm_eps)
        self.tf = nn.Sequential(self.tf_dense, self.tf_act, self.tf_ln)
        # self.decoder = BertOnlyMLMHead(config)
        # decoder
        self.decoder = nn.Linear(emb_dim, config.vocab_size, bias=False)
        self.loss_fct = CrossEntropyLoss(reduction="none")  # -100 index = padding token

    def forward(self, inputs):
        labels = inputs["labels"]
        feats = inputs["latent_tokens"]
        backward_indexes = inputs["backward_indexes"]
        backward_indexes = torch.cat(
            [
                torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                backward_indexes + 1,
            ],
            dim=0,
        )
        feats = torch.cat(
            [
                feats,
                self.mask_token.expand(
                    backward_indexes.shape[0] - feats.shape[0], feats.shape[1], -1
                ),
            ],
            dim=0,
        )
        feats = take_indexes(feats, backward_indexes)
        feats = feats + self.pos_embedding
        # check whether number of tokens between image and text match
        if labels.shape[1] != feats.shape[0]:
            assert (
                labels.shape[1] > feats.shape[0]
            ), "text sequence must be longer than number of image tokens"
            n_out = labels.shape[1]
            n_in = feats.shape[0]
            n_pad_tokens = n_out - n_in  # +1 because of CLS token
            pad_tokens = self.pad_token.expand(n_pad_tokens, feats.shape[1], -1)
            feats = torch.cat([feats, pad_tokens], dim=0)

        # transformation
        feats = self.tf(feats)
        prediction_scores = self.decoder(feats)
        masked_lm_loss = self.loss_fct(
            prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
        )
        masked_lm_loss = masked_lm_loss.view(labels.shape[0], labels.shape[1])
        return masked_lm_loss


class TFVisionToTextDecoder(pl.LightningModule):
    """
    Transformer Decoder (inspired by Tschannen et al., "Image Captioners Are Scalable Vision Learners Too", Neurips 2023
    """

    def __init__(
        self,
        emb_dim: int,
        n_dec_heads: int,
        mlp_ratio: int,
        qk_norm: bool,
        n_dec_layers: int,
        image_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        config = BertConfig()
        config.hidden_size = self.emb_dim
        self.config = config
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pad_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        # decoder
        self.decoder = torch.nn.Sequential(
            *[
                Block(
                    emb_dim,
                    n_dec_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_norm=qk_norm,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
                for _ in range(n_dec_layers)
            ]
        )
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, config.vocab_size, bias=False)
        self.loss_fct = CrossEntropyLoss(reduction="none")  # -100 index = padding token
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim)
        )

    def forward(self, inputs):
        labels = inputs["labels"]
        feats = inputs["latent_tokens"]
        backward_indexes = inputs["backward_indexes"]
        backward_indexes = torch.cat(
            [
                torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                backward_indexes + 1,
            ],
            dim=0,
        )
        feats = torch.cat(
            [
                feats,
                self.mask_token.expand(
                    backward_indexes.shape[0] - feats.shape[0], feats.shape[1], -1
                ),
            ],
            dim=0,
        )
        feats = take_indexes(feats, backward_indexes)
        feats = feats + self.pos_embedding
        # check whether number of tokens between image and text match
        if labels.shape[1] != feats.shape[0]:
            assert (
                labels.shape[1] > feats.shape[0]
            ), f"text sequence must be longer than number of image tokens: len(text): {labels.shape}, len(img): {feats.shape}"
            n_out = labels.shape[1]
            n_in = feats.shape[0]
            n_pad_tokens = n_out - n_in + 1  # +1 because of CLS token
            pad_tokens = self.pad_token.expand(n_pad_tokens, feats.shape[1], -1)
            feats = torch.cat([feats, pad_tokens], dim=0)

        # transformation
        feats = rearrange(feats, "t b c -> b t c")
        feats = self.decoder(feats)
        feats = self.layer_norm(feats)
        feats = rearrange(feats, "b t c -> t b c")
        feats = feats[1:]
        prediction_scores = self.out_head(feats)
        masked_lm_loss = self.loss_fct(
            prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
        )
        masked_lm_loss = masked_lm_loss.view(labels.shape[0], labels.shape[1])
        return masked_lm_loss


class CapPaVisionToTextDecoder(pl.LightningModule):
    """
    Transformer Decoder (inspired by Tschannen et al., "Image Captioners Are Scalable Vision Learners Too", Neurips 2023
    """

    def __init__(
        self,
        emb_dim: int,
        n_dec_heads: int,
        mlp_ratio: int,
        n_dec_layers: int,
        n_img_patches: int,
        tokenizer_config: TokenizerModuleConfig,
        enable_parallel_decoding: bool,
        cappa_per_view: bool,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cappa_per_view = cappa_per_view
        self.enable_parallel_decoding = enable_parallel_decoding
        self.tokenizer_config = tokenizer_config

        if self.tokenizer_config is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_config.pretrained_model_name
            )
            self.max_len = self.tokenizer_config.max_len
            self.truncation = self.tokenizer_config.truncation
            if self.tokenizer_config.dec_token:
                self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.emb_dim = emb_dim
        config = BertConfig()
        config.hidden_size = self.emb_dim
        self.config = config
        self.inp_vocabulary = torch.nn.Embedding(config.vocab_size, emb_dim)
        # text decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=n_dec_heads,
            dim_feedforward=emb_dim * mlp_ratio,
            activation="gelu",
            bias=False,
        )
        layer_norm = torch.nn.LayerNorm(emb_dim)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_dec_layers, norm=layer_norm
        )
        self.out_head = nn.Linear(emb_dim, config.vocab_size, bias=False)
        self.loss_fct = CrossEntropyLoss(reduction="none")  # -100 index = padding token
        self.pos_embedding = torch.nn.Parameter(torch.zeros(n_img_patches, 1, emb_dim))
        self.mask_embedding = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))

    def get_start_id(self):
        start_id = (
            getattr(self.tokenizer, "bos_token_id", None)
            or getattr(self.tokenizer, "cls_token_id", None)
            or getattr(self.tokenizer, "pad_token_id", 0)
        )
        return start_id

    def generate(self, inputs, n_tokens=100):
        feats = self.prepare_img_tokens(inputs)
        batch_size = feats.shape[1]
        # determine start token id
        start_id = self.get_start_id()
        eos_id = getattr(self.tokenizer, "eos_token_id", None)

        # initialize generated ids and embeddings: shape (t, b)
        gen_ids = torch.full(
            (1, batch_size), start_id, dtype=torch.long, device=feats.device
        )
        gen_emb = self.inp_vocabulary(gen_ids.view(-1)).view(1, batch_size, -1)

        for _step in range(n_tokens):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                gen_emb.size(0)
            ).to(feats.device)
            out_dec = self.decoder(
                gen_emb, feats, tgt_mask=tgt_mask, tgt_is_causal=True
            )
            last_hidden = out_dec[-1]  # (b, emb_dim)
            logits = self.out_head(last_hidden)  # (b, vocab)
            next_tokens = logits.argmax(dim=-1)  # (b,)

            gen_ids = torch.cat([gen_ids, next_tokens[None]], dim=0)
            next_emb = self.inp_vocabulary(next_tokens)  # (b, emb_dim)
            gen_emb = torch.cat([gen_emb, next_emb[None]], dim=0)

            if eos_id is not None and (next_tokens == eos_id).all():
                break

        # decode generated ids to strings
        gen_ids_T = gen_ids.T.cpu().numpy().tolist()
        gen_texts = self.tokenizer.batch_decode(gen_ids_T, skip_special_tokens=True)

        return gen_texts

    def _prepare_img_tokens(self, inputs):
        feats = inputs["latent_tokens"]
        backward_indexes = inputs["backward_indexes"][
            : feats.shape[0] - 1
        ]  # Get original idx of vision tokens
        backward_indexes = torch.cat(
            [
                torch.zeros(1, backward_indexes.shape[1]).to(
                    backward_indexes
                ),  # Add CLS token idx (first index of self.pos_embedding)
                backward_indexes + 1,
            ],
            dim=0,
        )
        pos_embeddings = take_indexes(
            self.pos_embedding.expand(-1, backward_indexes.shape[1], -1),
            backward_indexes,
        )
        feats = feats + pos_embeddings
        return feats

    def prepare_img_tokens(self, inputs):
        if self.cappa_per_view:
            return self._prepare_img_tokens(inputs)
        else:
            latent_tokens = inputs["latent_tokens"]
            backward_indexes = inputs["backward_indexes"]
            all_feats = []
            for tokens, indexes in zip(latent_tokens, backward_indexes):
                view_inputs = {
                    "latent_tokens": tokens,
                    "backward_indexes": indexes,
                }
                view_feats = self._prepare_img_tokens(view_inputs)
                all_feats.append(view_feats)
            return torch.cat(all_feats, dim=0)

    def prediction_to_text(self, prediction_scores, labels):
        text_preds = prediction_scores.argmax(dim=-1).T
        text_gts = labels.T
        text_preds[text_gts == -100] = self.tokenizer.pad_token_id
        text_gts[text_gts == -100] = self.tokenizer.pad_token_id
        text_preds = self.tokenizer.batch_decode(
            text_preds, skip_special_tokens=True
        )  # list of strings
        text_gts = self.tokenizer.batch_decode(
            text_gts, skip_special_tokens=True
        )  # list of strings
        return text_preds, text_gts

    def forward(self, inputs):
        labels = inputs["labels"]
        labels[..., 0] = self.get_start_id()
        feats = self.prepare_img_tokens(inputs)

        if not self.enable_parallel_decoding or random.random() < 0.5:
            text_embedding = torch.zeros(
                labels.shape[0], labels.shape[1], self.emb_dim
            ).to(feats.device)
            text_embedding[labels != -100] = self.inp_vocabulary(
                labels[labels != -100]
            )  # Add text embeddings for unpadded inputs
            text_embedding = text_embedding.permute(1, 0, 2)

            # transformer decoder with cross-attention
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                text_embedding.shape[0]
            ).to(feats.device)
            preds = self.decoder(
                text_embedding, feats, tgt_mask=causal_mask, tgt_is_causal=True
            )
        else:
            # Parallel decoding (non-autoregressive)
            # Use MASK token for all text positions
            text_embedding = self.mask_embedding.expand(
                labels.shape[1], labels.shape[0], -1
            )  # (t, b, emb_dim)
            # transformer decoder with cross-attention
            preds = self.decoder(
                text_embedding, feats, tgt_is_causal=False
            )  # No causal mask
        # Align predictions and labels for next token prediction
        preds = preds[:-1]
        labels = labels.T[1:]
        prediction_scores = self.out_head(preds).reshape(-1, self.config.vocab_size)
        masked_lm_loss = self.loss_fct(prediction_scores, labels.reshape(-1))
        masked_lm_loss = masked_lm_loss.view(labels.shape[0], labels.shape[1]).sum(
            dim=0
        ) / (labels != -100).sum(
            dim=0
        )  # Average over tokens so final loss shape is (batch_size,)
        # Get ground truth and predictions as text
        prediction_scores = prediction_scores.reshape(
            preds.shape[0], preds.shape[1], -1
        )
        text_preds, text_gts = self.prediction_to_text(prediction_scores, labels)
        return masked_lm_loss, text_preds, text_gts