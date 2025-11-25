import random

import torch
import torch.nn as nn
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from PIL import Image, ImageDraw
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from functools import partial

from torchvision.transforms.functional import to_pil_image


# code taken from https://github.com/lambert-x/medical_mae
# https://github.com/IcarusWizard/MAE


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(
        sequences, 0, repeat(indexes, "t b -> t b c", c=sequences.shape[-1])
    )


class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(
            np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)
        backward_indexes = torch.as_tensor(
            np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes[:remain_T], backward_indexes


class MAE_Encoder(torch.nn.Module):
    def __init__(
        self,
        patch_size: int,
        mask_ratio: float,
        emb_dim: int,
        n_enc_heads: int,
        n_enc_layers: int,
        image_size: int,
        num_channels: int,
        mlp_ratio: int,
        qk_norm: bool,
        mod_embedding: int,
        log_debug: bool,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros((image_size // patch_size) ** 2, 1, emb_dim)
        )
        self.use_modality_emb = mod_embedding > 0
        if self.use_modality_emb:
            self.mod_embedding = torch.nn.Parameter(torch.zeros(mod_embedding, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(num_channels, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(
            *[
                Block(
                    emb_dim,
                    n_enc_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_norm=qk_norm,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
                for _ in range(n_enc_layers)
            ]
        )

        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.emb_dim = emb_dim
        self.image_size = image_size
        self.log_debug = log_debug

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)
        if self.use_modality_emb:
            trunc_normal_(self.mod_embedding, std=0.02)

    def forward(self, inputs):
        img = inputs["data"]
        patches = self.patchify(img)
        patches = rearrange(patches, "b c h w -> (h w) b c")
        patches = patches + self.pos_embedding

        if self.use_modality_emb:
            modality_index = inputs["modality_index"]
            patches += self.mod_embedding[modality_index]

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat(
            [self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0
        )
        patches = rearrange(patches, "t b c -> b t c")
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, "b t c -> t b c")

        count_non_informative_patches = 0
        count_non_informative_patches_img_max = 0

        if self.log_debug:
            for b in range(img.shape[0]):
                img_b = img[b]  # shape: [C, H, W]
                visible_indexes_b = forward_indexes[:, b]  # shape: [T_keep]
                count_non_informative_patches_img = visualize_visible_patches_pil(
                    img_b, visible_indexes_b, patch_size=self.patch_size
                )
                if (
                    count_non_informative_patches_img
                    > count_non_informative_patches_img_max
                ):
                    count_non_informative_patches_img_max = (
                        count_non_informative_patches_img
                    )
                count_non_informative_patches += count_non_informative_patches_img

            return {
                "features": features,
                "backward_indexes": backward_indexes,
                "bad_patches_ratio": count_non_informative_patches
                / (backward_indexes.shape[0] * backward_indexes.shape[1]),
                "max_bad_patches_imgs": count_non_informative_patches_img,
            }
        else:
            return {"features": features, "backward_indexes": backward_indexes}

    def set_mask_ratio(self, ratio):
        self.shuffle = PatchShuffle(ratio)


def visualize_visible_patches_pil(
    img_tensor, forward_indexes, patch_size=16, save_path=None
):
    img_h, img_w = img_tensor.shape[-2:]
    grid_w = img_w // patch_size

    img_pil = to_pil_image(img_tensor.squeeze(0).cpu()).convert(
        "RGB"
    )  # to display grid color
    draw = ImageDraw.Draw(img_pil)

    count_non_informative_patches = 0

    for idx in forward_indexes:
        idx = idx.item()
        row = idx // grid_w
        col = idx % grid_w
        top_left = (col * patch_size, row * patch_size)
        bottom_right = ((col + 1) * patch_size - 1, (row + 1) * patch_size - 1)

        top = row * patch_size
        left = col * patch_size
        bottom = top + patch_size
        right = left + patch_size

        patch = img_tensor[:, top:bottom, left:right]

        flat_patch = patch.flatten()

        tolerance = 1e-2

        # Compare to first pixel
        ref_val = flat_patch[0]
        equal_mask = torch.abs(flat_patch - ref_val) < tolerance
        percent_equal = equal_mask.float().mean().item()

        # Threshold: if more than 90% of pixels are equal -> mark as red
        uniform_threshold = 0.9
        if percent_equal > uniform_threshold:
            color = "red"
            count_non_informative_patches += 1
        else:
            color = "green"

        draw.rectangle([top_left, bottom_right], outline=color, width=1)

    if save_path and count_non_informative_patches > 7:
        random_suffix = random.randint(10, 15)
        filename_parts = save_path.split(".")
        save_path = f"{filename_parts[0]}_{random_suffix}.{filename_parts[1]}"
        img_pil.save(save_path)

    return count_non_informative_patches


class MAE_Decoder(torch.nn.Module):
    def __init__(
        self,
        patch_size: int,
        emb_dim: int,
        n_dec_heads: int,
        n_dec_layers: int,
        image_size: int,
        num_channels: int,
        mlp_ratio: int,
        qk_norm: bool,
        mod_embedding: int,
        use_layer_norm: bool,
    ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim)
        )
        self.use_modality_emb = mod_embedding > 0
        if self.use_modality_emb:
            self.mod_embedding = torch.nn.Parameter(torch.zeros(mod_embedding, emb_dim))

        self.transformer = torch.nn.Sequential(
            *[
                Block(
                    emb_dim,
                    n_dec_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    # qk_scale=None,
                    qk_norm=qk_norm,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
                for _ in range(n_dec_layers)
            ]
        )
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.head = torch.nn.Linear(emb_dim, num_channels * patch_size**2)
        self.patch2img = Rearrange(
            "(h w) b (c p1 p2) -> b c (h p1) (w p2)",
            p1=patch_size,
            p2=patch_size,
            h=image_size // patch_size,
        )

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)
        if self.use_modality_emb:
            trunc_normal_(self.mod_embedding, std=0.02)

    def forward(self, inputs):
        features = inputs["features"]
        backward_indexes = inputs["backward_indexes"]
        T = features.shape[0]
        backward_indexes = torch.cat(
            [
                torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes),
                backward_indexes + 1,
            ],
            dim=0,
        )
        features = torch.cat(
            [
                features,
                self.mask_token.expand(
                    backward_indexes.shape[0] - features.shape[0], features.shape[1], -1
                ),
            ],
            dim=0,
        )
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding
        if self.use_modality_emb:
            modality_index = inputs["modality_index"]
            features += self.mod_embedding[modality_index]

        features = rearrange(features, "t b c -> b t c")
        features = self.transformer(features)
        if self.use_layer_norm:
            features = self.layer_norm(features)

        features = rearrange(features, "b t c -> t b c")
        features = features[1:]  # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T - 1 :] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        # return {"img":img, "mask":mask}
        return {"img": img, "mask": mask}
