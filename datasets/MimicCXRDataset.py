import os

# from open_clip import create_model_from_pretrained
from typing import Dict, Any
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.generic import (
    get_text_modalities,
)
import ast
from transformers import AutoProcessor


def extract_normalize_info(transform):
    if isinstance(transform, v2.Compose):
        rgb = 1
        norm = False
        param = None
        for t in transform.transforms:
            if (
                isinstance(t, v2.Grayscale)
                and getattr(t, "num_output_channels", 1) == 3
            ):
                rgb = 3
            if isinstance(t, v2.Normalize):
                norm, param = True, {"mean": t.mean, "std": t.std}
    elif transform.__name__ == "CheXagentProcessor":
        rgb = 3
        norm = None
        param = None
    return rgb, norm, param


def create_labels(metadata, label_names) -> pd.DataFrame:
    labels = metadata[label_names].copy()
    labels = labels.fillna(0)
    labels.replace(-1, 0, inplace=True)
    assert all((labels == 0) | (labels == 1))
    return labels


def merge_and_get_metadata_split(
    fn_metadata, fn_metadata_text, split, debug, num_train_sample=0
):
    metadata = pd.read_csv(fn_metadata)
    metadata_text = pd.read_csv(fn_metadata_text)

    # Process 'study' column in fn_metadata_text to match 'study_id' format (no 's' in the beginning)
    metadata_text["study_id"] = metadata_text["study"].str[1:].astype(int)

    # Check if "impression_plus_findings" is available for backward compatitibility
    if "impression_plus_findings" in metadata_text.columns:
        metadata = metadata.merge(
            metadata_text[
                ["study_id", "findings", "impression", "impression_plus_findings"]
            ],
            on="study_id",
            how="left",
        )
    else:
        # Merge metadata_text into metadata based on study_id
        metadata = metadata.merge(
            metadata_text[["study_id", "findings", "impression"]],
            on="study_id",
            how="left",
        )

    # Filter by split
    metadata_split = metadata[metadata["split"] == split]

    if split != "train" and debug:
        raise Exception("DEBUG SHOULD BE SET TO TRUE ONLY ON THE TRAINING SET")

    if debug:
        metadata_split = metadata_split[:num_train_sample]

    return metadata_split


def get_metadata_split(fn_metadata, split, debug, num_train_sample=0):
    metadata = pd.read_csv(fn_metadata)
    if split != "train" and debug:
        raise Exception("DEBUG SHOULD BE SET TO TRUE ONLY ON THE TRAINING SET")
    if debug:
        metadata = metadata[:num_train_sample]
    metadata_split = metadata[metadata["split"] == split]
    return metadata_split


def get_transform_folder_mae(
    img_size: int,
    resize_crop: bool,
    resize_crop_kwargs: Dict[str, Any],
    rotation: bool,
    rotation_kwargs: Dict[str, Any],
    horizontal_flip: bool,
    normalize: bool,
    normalize_kwargs: Dict[str, Any],
    jitter: bool,
    jitter_kwargs: Dict[str, Any],
    rgb: bool,
):
    tfs = []
    if not rgb:
        tfs.append(v2.Grayscale(num_output_channels=1))
    else:
        tfs.append(v2.Grayscale(num_output_channels=3))
    if resize_crop:
        tfs.append(v2.RandomResizedCrop(**resize_crop_kwargs))
    else:
        tfs.append(v2.Resize(size=(img_size, img_size)))
    if rotation:
        tfs.append(v2.RandomRotation(**rotation_kwargs))
    if horizontal_flip:
        tfs.append(v2.RandomHorizontalFlip())
    if jitter:
        tfs.append(v2.RandomApply([v2.ColorJitter(**jitter_kwargs)], p=0.8))
    tfs.append(v2.ToImage())
    tfs.append(v2.ToDtype(torch.float32, scale=True))
    if normalize:
        tfs.append(v2.Normalize(**normalize_kwargs))
    tf = v2.Compose(tfs)
    return tf


class ProcessorTransform:
    def __init__(self, processor):
        self.processor = processor
        self.__name__ = "CheXagentProcessor"

    def __call__(self, img):
        img = transforms.ToPILImage()(img)
        img = img.convert("L")  # Ensure grayscale
        processed = self.processor(images=img, return_tensors="pt")
        return processed["pixel_values"].squeeze()


def get_transform_folder_stanfordvit(cache_dir):
    processor = AutoProcessor.from_pretrained(
        "StanfordAIMI/CheXagent-8b", trust_remote_code=True, cache_dir=cache_dir
    )
    processor_transform = ProcessorTransform(processor)
    return processor_transform


# support refactored metadata - multi dataset
class CXRDataset(Dataset):
    def __init__(
        self,
        vision_modality_config,
        text_modality_config,
        transform,
        dir_data,
        img_size,
        target_list,
        split,
        debug,
        num_train_sample,
        vision_text_reconstruction,
        tokenizer,
        split_seed,
        version,
        studies_policy,
        fn_metadata,
        selected_datasets=None,  # if None, all datasets are used, otherwise only the selected dataset is used.
    ):
        self.num_train_sample = num_train_sample
        self.dir_data = dir_data
        self.img_size = img_size
        self.transform = transform
        self.rgb, _, _ = extract_normalize_info(self.transform)
        self.split = split
        self.label_names = target_list
        self.vision_modality_config = vision_modality_config
        self.text_modality_config = text_modality_config
        self.split_seed = split_seed
        self.version = version
        self.studies_policy = studies_policy
        self.selected_datasets = selected_datasets
        self.vision_text_reconstruction = None
        if self.text_modality_config is not None:
            self.input_text_modalities = False
        else:
            self.input_text_modalities = True

        self.text_modalities = get_text_modalities(
            self.text_modality_config, self.vision_modality_config
        )
        print(self.text_modalities)

        if self.studies_policy == "all_combi_no_missing":
            self.get_vision_views = self.get_vision_views_acnm
        elif self.studies_policy == "FL" or self.studies_policy == "FLU":
            self.get_vision_views = self.get_vision_views_structured
        elif self.studies_policy == "all":
            self.get_vision_views = self.get_vision_views_all
        elif (
            self.studies_policy == "FLU_DETERMINISTIC"
            or self.studies_policy == "FL_DETERMINISTIC"
        ):
            self.get_vision_views = self.get_vision_views_structured_deterministic

        # metadata loading
        self.metadata = pd.read_csv(fn_metadata, low_memory=False)
        if selected_datasets is not None:
            # filter metadata to only include the selected datasets
            self.metadata = self.metadata[
                self.metadata["dataset"].isin(selected_datasets)
            ]
            if self.metadata.empty:
                raise ValueError(
                    f"No data found for the selected datasets: {selected_datasets}"
                )
            # assert all the selected datasets are in the metadata
            assert all(
                ds in self.metadata["dataset"].values for ds in selected_datasets
            )

        # shuffle metadata if split is train
        if split == "train":
            self.metadata = self.metadata.sample(
                frac=1, random_state=self.split_seed
            ).reset_index(drop=True)

        # prepare metadata
        self.prep_metadata = prepare_metadata(
            self.metadata, self.studies_policy, debug, self.num_train_sample
        )

        # Tokenizer Setup
        if tokenizer:
            self.tokenizer_config = tokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_config.pretrained_model_name
            )
            self.max_len = self.tokenizer_config.max_len
            self.truncation = self.tokenizer_config.truncation
            if self.tokenizer_config.dec_token:
                self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            if self.tokenizer_config.vlm:
                self.tokenizer_vlm = AutoTokenizer.from_pretrained(
                    self.tokenizer_config.pretrained_model_name_vlm, use_fast=True
                )
                self.tokenizer_vlm.pad_token = self.tokenizer_vlm.eos_token

            # Text Modalities Processing
            self.ids_default, self.attention_mask_default = self.encode_lines(
                ["default"]
            )
            self.ids, self.ids_vlm = {}, {}
            self.attention_mask, self.attention_mask_vlm = {}, {}
            text_modalities_pbar = tqdm(self.text_modalities)
            for tm in text_modalities_pbar:
                text_modalities_pbar.set_description("Processing " + tm + " " + split)
                if tm in self.prep_metadata.columns:
                    self.ids[tm], self.attention_mask[tm] = self.encode_lines(
                        self.prep_metadata[tm].fillna("").tolist()
                    )
                    if self.tokenizer_config.vlm:
                        self.ids_vlm[tm], self.attention_mask_vlm[tm] = (
                            self.encode_lines(
                                self.prep_metadata[tm].fillna("").tolist(), tok=False
                            )
                        )
                else:
                    raise Exception("Requested Text Modality not available in dataset")
        else:
            print("no tokenizer provided")

        self.labels = create_labels(self.prep_metadata, self.label_names)
        print("Dataset Init Completed")
        print("split: " + split)
        print("selected datasets: " + str(selected_datasets) + "\n")
        # print number of studies per dataset

        for ds in set(self.metadata["dataset"].values):
            num_studies = self.metadata[self.metadata["dataset"] == ds].shape[0]
            print(f" - {ds}: {num_studies} studies")

    def encode_lines(self, lines, tok=True):
        if tok:
            tokenizer = self.tokenizer
        else:  # Only for the VLM to add extra tokenizer
            tokenizer = self.tokenizer_vlm
        batch_encoding = tokenizer(
            lines,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=self.truncation,
            padding="max_length",
        )
        return batch_encoding["input_ids"], batch_encoding["attention_mask"]

    def __len__(self):
        return len(self.labels)

    def sample_img_fn(
        self, filenames: list[str], views: list[str], target_view: str
    ) -> str:
        """
        Args:
            filenames (list of str): List of image filenames.
            views (list of str): List of corresponding view labels.
            view (str): View label to sample from (e.g., 'F' or 'L').

        Returns:
            str: Randomly selected filename for the specified view.

        Raises:
            TypeError: If input types are incorrect.
            ValueError: If lengths mismatch or no match found.
        """
        if not isinstance(filenames, list) or not isinstance(views, list):
            raise TypeError("Expected both filenames and views to be lists.")

        if len(filenames) != len(views):
            raise ValueError(
                f"Mismatched lengths: {len(filenames)} filenames vs {len(views)} views."
            )

        filenames_np = np.array(filenames)
        views_np = np.array(views)

        match_mask = views_np == target_view
        if not np.any(match_mask):
            raise ValueError(f"No filenames found for view '{target_view}'.")

        matching_files = filenames_np[match_mask]
        return matching_files[np.random.randint(len(matching_files))], target_view

    def sample_img_fn_num_views(
        self,
        filenames: list[str],
        views: list[str],
        n_views: int,
        max_num_views: int,
        view_policy: str,
    ):
        filenames_np = np.array(filenames)
        views_np = np.array(views)
        if view_policy == "FL":
            # only use F and L views, discard U
            filter_v = np.where(views_np != "U")[0]
            views_np = views_np[filter_v]
            filenames_np = filenames_np[filter_v]

        if len(filenames_np) > max_num_views:
            idx = np.random.choice(len(filenames_np), size=max_num_views, replace=False)
            views_np = views_np[idx]
            filenames_np = filenames_np[idx]
        return filenames_np, views_np

    def deterministic_selection_img_fn_num_views(
        self,
        filenames: list[str],
        views: list[str],
        n_views: int,
        max_num_views: int,
        view_policy: str,
    ):
        filenames_np = np.array(filenames)
        views_np = np.array(views)
        if view_policy == "FL_DETERMINISTIC":
            # only use F and L views, discard U
            filter_v = np.where(views_np != "U")[0]
            views_np = views_np[filter_v]
            filenames_np = filenames_np[filter_v]

        if len(filenames_np) > max_num_views:
            # Take the first max_num_views elements deterministically
            views_np = views_np[:max_num_views]
            filenames_np = filenames_np[:max_num_views]
        return filenames_np, views_np

    def get_vision_views_acnm(self, metadata):
        vision_dict = {}
        for key in self.vision_modality_config.views:
            fn_v, view_v = self.sample_img_fn(
                metadata["filenames"], metadata["views"], key
            )
            fn_v = os.path.join(self.dir_data, metadata["dataset"], fn_v)
            v_avail = False
            v_data = torch.zeros([self.rgb, self.img_size, self.img_size])
            if os.path.exists(fn_v):
                img_v = read_image(fn_v)
                img_v = self.transform(img_v)
                v_data = img_v
                v_avail = True
            else:
                print(f"WARNING LOADING {key}: {fn_v}")
            view_dict = {
                "data": v_data,
                "mask": v_avail,
                "index": self.vision_modality_config["view_indices"][key],
            }
            vision_dict[key] = view_dict
        return vision_dict

    def generate_vision_dict(self, metadata, filenames, views):
        max_n_views = self.vision_modality_config.max_num_views
        vision_dict = {}
        for v_idx in range(0, max_n_views):
            v_avail = False
            v_data = torch.zeros([self.rgb, self.img_size, self.img_size])
            v_index = 0
            if len(filenames) > v_idx:
                key_v = self.vision_modality_config.view_indices[views[v_idx]]
                fn_v = os.path.join(
                    self.dir_data, metadata["dataset"], filenames[v_idx]
                )
                if os.path.exists(fn_v):
                    img_v = read_image(fn_v)
                    img_v = self.transform(img_v)
                    v_data = img_v
                    v_avail = True
                    v_index = key_v
                else:
                    print(f"WARNING LOADING {views[v_idx]}: {fn_v}")
            view_dict = {"data": v_data, "mask": v_avail, "index": v_index}
            vision_dict[f"v{v_idx}"] = view_dict
        return vision_dict

    def get_vision_views_structured(self, metadata):
        max_n_views = self.vision_modality_config.max_num_views
        fns, views = self.sample_img_fn_num_views(
            metadata["filenames"],
            metadata["views"],
            metadata["n_views"],
            max_n_views,
            view_policy=self.studies_policy,
        )
        vision_dict = self.generate_vision_dict(metadata, fns, views)
        return vision_dict

    def get_vision_views_all(self, metadata):
        max_num_views = self.vision_modality_config.max_num_views
        fns, views = self.sample_img_fn_num_views(
            metadata["filenames"],
            metadata["view_positions"],
            metadata["n_views"],
            max_num_views,
            view_policy=self.studies_policy,
        )
        vision_dict = self.generate_vision_dict(metadata, fns, views)
        return vision_dict

    def get_vision_views_structured_deterministic(self, metadata):
        max_n_views = self.vision_modality_config.max_num_views
        fns, views = self.deterministic_selection_img_fn_num_views(
            metadata["filenames"],
            metadata["views"],
            metadata["n_views"],
            max_n_views,
            view_policy=self.studies_policy,
        )
        vision_dict = self.generate_vision_dict(metadata, fns, views)
        return vision_dict

    def __getitem__(self, index):
        label_values = self.labels.iloc[index]
        metadata = self.prep_metadata.iloc[index]

        sample_dict = {}
        # vision modality
        if self.vision_modality_config is not None:
            vision_dict = self.get_vision_views(metadata)
            sample_dict["vision"] = vision_dict

        # text modality
        if len(self.text_modalities) > 0:
            text_dict = {}
            for tm in self.text_modalities:
                v_data = {
                    "input_ids": self.ids_default[0],
                    "attention_mask": self.attention_mask_default[0],
                }
                if self.tokenizer_config.vlm:
                    v_data.update(
                        {
                            "input_ids_vlm": self.ids_default[0],
                            "attention_mask_vlm": self.attention_mask_default[0],
                        }
                    )

                v_avail = False
                if tm in metadata and pd.notna(metadata[tm]) and metadata[tm].strip():
                    v_data = {
                        "input_ids": self.ids[tm][index],
                        "attention_mask": self.attention_mask[tm][index],
                    }
                    if self.tokenizer_config.vlm:
                        v_data.update(
                            {
                                "input_ids_vlm": self.ids_default[0],
                                "attention_mask_vlm": self.attention_mask_default[0],
                            }
                        )
                    v_avail = True
                if self.text_modality_config is not None:
                    v_index = self.text_modality_config["view_indices"][tm]
                else:
                    v_index = 0
                view_dict = {"data": v_data, "mask": v_avail, "index": v_index}

                text_dict[tm] = view_dict
            sample_dict["text"] = text_dict

        label = torch.tensor(label_values.values.astype(int)).float()
        return sample_dict, label


def prepare_metadata(metadata, study_policy, debug, num_train_sample):
    """
    Prepares the metadata based on the study policy.
    Args:
        metadata (pd.DataFrame): The metadata DataFrame.
        study_policy (str): The study policy to apply.
    Returns:
        pd.DataFrame: The prepared metadata DataFrame.
    """
    # check that views, view_positions, filenames can be read as list and not as string
    metadata["views"] = metadata["views"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    metadata["view_positions"] = metadata["view_positions"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    metadata["filenames"] = metadata["filenames"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # manipulation
    print("study policy: ", study_policy)
    if study_policy == "all_combi_no_missing":
        # print numbers before and after
        print("Study Metadata before policy filtering:", len(metadata))
        # filter out all the studies with n_views < 2
        metadata = metadata[metadata["n_views"] >= 2]
        # keep all the studies with at list one "F" and at least on "L" in views column
        metadata = metadata[metadata["views"].apply(lambda v: "F" in v and "L" in v)]
        # print numbers after
        print("Study Metadata after policy filtering:", len(metadata))
    if study_policy == "FL":
        # print numbers before and after
        print("Study Metadata before policy filtering:", len(metadata))
        # keep all the studies with at least one "F" or at least on "L" in views column
        metadata = metadata[metadata["views"].apply(lambda v: "F" in v or "L" in v)]
        # print numbers after
        print("Study Metadata after policy filtering:", len(metadata))
    if study_policy == "FLU":
        # print numbers before and after
        print("Study Metadata before policy filtering:", len(metadata))
        # keep all the studies with at least one "F" or at least on "L" in views column
        metadata = metadata[
            metadata["views"].apply(lambda v: "F" in v or "L" in v or "U" in v)
        ]
        # print numbers after
        print("Study Metadata after policy filtering:", len(metadata))
    if study_policy == "FLU_DETERMINISTIC":
        # print numbers before and after
        print("Study Metadata before policy filtering:", len(metadata))
        metadata = metadata[
            metadata["views"].apply(lambda v: "F" in v or "L" in v or "U" in v)
        ]
        num_views_study_filter = 2

        # fileter out all the studies with n_views != num_views_study_filter
        if num_views_study_filter is not None:
            metadata = metadata[metadata["n_views"] == num_views_study_filter]
        print("Study Metadata after policy filtering:", len(metadata))
    if debug:
        metadata = metadata[:num_train_sample]

    metadata["filenames"] = metadata["filenames"].apply(
        lambda x: (
            [fn.replace("mimic-cxr/", "") for fn in x] if isinstance(x, list) else x
        )
    )

    return metadata
