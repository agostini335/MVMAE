from typing import List

import torch
from einops import rearrange
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.data.DatasetConfig import TextModConfig, VisionModConfig

IMPLEMENTED_METRICS = [
    "AP",
    "AUROC",
    "F1",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
]


class MimicEvalBase:
    def __init__(
        self,
        vision_modality_config: VisionModConfig,
        text_modality_config: TextModConfig,
        max_iter: int,
        evaluate_lr: bool,
        logging_frequency_lr: int,
        logging_frequency_plots: int,
        clfs: List[str],
        metrics: List[str],
        f_n_jobs: int,
        seed: int,
    ) -> None:
        self.seed = seed
        self.vision_modality_config = vision_modality_config
        self.text_modality_config = text_modality_config
        self.max_iteration = max_iter
        self.evaluate_lr = evaluate_lr
        self.logging_frequency_lr = logging_frequency_lr
        self.logging_frequency_plots = logging_frequency_plots
        self.clfs = clfs
        self.metrics = metrics
        self.f_n_jobs = f_n_jobs
        self.auroc_per_dataset = []

    def train_clfs(self, encodings, labels):
        pass

    def eval_clfs(self, clfs_dict, encodings, labels, device):
        # check metrics availability
        for metric in self.metrics:
            if metric not in IMPLEMENTED_METRICS:
                raise NotImplementedError(metric)

        n_labels = labels.shape[1]
        clfs_scores = {}

        for clf_name, trained_clfs in clfs_dict.items():
            # initialize scores as a dict with metrics as keys
            # example: scores['AUROC'][0] return the auroc score for label nr 0
            scores = {}
            for metric in self.metrics:
                scores[metric] = torch.zeros(n_labels, device=device)

            for k in range(0, n_labels):
                clf_k = trained_clfs[k]
                y_prob_k = clf_k.predict_proba(encodings)[:, 1]
                y_pred = clf_k.predict(encodings)
                if "AP" in self.metrics:
                    scores["AP"][k] = average_precision_score(labels[:, k], y_prob_k)
                if "AUROC" in self.metrics:
                    try:
                        scores["AUROC"][k] = roc_auc_score(labels[:, k], y_prob_k)
                    except ValueError:
                        scores["AUROC"][k] = 0.0
                if "F1" in self.metrics:
                    scores["F1"][k] = f1_score(labels[:, k], y_pred, zero_division=1.0)
                if "accuracy" in self.metrics:
                    scores["accuracy"][k] = accuracy_score(labels[:, k], y_pred)
                if "balanced_accuracy" in self.metrics:
                    scores["balanced_accuracy"][k] = balanced_accuracy_score(
                        labels[:, k], y_pred
                    )
                if "precision":
                    scores["precision"][k] = precision_score(
                        labels[:, k], y_pred, zero_division=1.0
                    )
                if "recall":
                    scores["recall"][k] = recall_score(
                        labels[:, k], y_pred, zero_division=1.0
                    )
            clfs_scores[clf_name] = scores
            if "AUROC" in self.metrics:
                self.auroc_per_dataset.append(scores["AUROC"].mean())
        return clfs_scores

    def train_clf_lr(self, encodings, labels):
        clf = LogisticRegression(max_iter=self.max_iteration).fit(encodings, labels)
        return clf

    def evaluate_conditional_generation(self, out, batch, model):
        return None, None

    def eval_conditional_generation(self, preds_coherence, labels_val):
        logs = {}
        return logs

    def eval_lr(
        self,
        emb_train,
        labels_train,
        emb_val,
        labels_val,
        device,
    ):
        clfs = {}
        # train linear clfs on representations that are fed into decoder
        for key in emb_train.keys():
            emb_m_train = emb_train[key]
            clf_out_m = self.train_clfs(
                emb_m_train,
                labels_train,
            )
            clfs[key] = clf_out_m

        # validate the linear classifiers
        # on for m, key in enumerate(self.modality_names):
        logs = {}
        for key in emb_train.keys():
            clf_m = clfs[key]
            emb_m_val = emb_val[key]
            scores_m = self.eval_clfs(
                clf_m,
                emb_m_val,
                labels_val,
                device,
            )
            # scores_m : RF -> AUROC -> Label
            for clf_type in self.clfs:
                for metric in self.metrics:
                    logs[f"downstream/{key}/{clf_type}/{metric}"] = scores_m[clf_type][
                        metric
                    ].mean()
        return logs

    def get_training_output(self, training_out):
        pass

    def get_validation_output(self, validation_out):
        pass

    def eval(self, model):
        pass


class MimicMAEEval(MimicEvalBase):
    def __init__(
        self,
        vision_modality_config: VisionModConfig,
        text_modality_config: TextModConfig,
        max_iter: int,
        evaluate_lr: bool,
        logging_frequency_lr: int,
        logging_frequency_plots: int,
        clfs: List[str],
        metrics: List[str],
        f_n_jobs: int,
        seed: int,
        f_n_estimators: int,
        f_min_samples_split: int,
        f_min_samples_leaf: int,
        f_max_features: str,
        f_max_depth: int,
        f_criterion: str,
        f_bootstrap: bool,
    ) -> None:
        super().__init__(
            vision_modality_config,
            text_modality_config,
            max_iter,
            evaluate_lr,
            logging_frequency_lr,
            logging_frequency_plots,
            clfs,
            metrics,
            f_n_jobs,
            seed,
        )
        self.f_n_estimators = f_n_estimators
        self.f_min_samples_split = f_min_samples_split
        self.f_min_samples_leaf = f_min_samples_leaf
        self.f_max_features = f_max_features
        self.f_max_depth = f_max_depth
        self.f_criterion = f_criterion
        self.f_bootstrap = f_bootstrap

    def train_clfs(self, encodings, labels):
        # check classifier availability
        for clf in self.clfs:
            if clf != "RF" and clf != "LR":
                raise NotImplementedError("Only RF and LR are supported")

        n_labels = labels.shape[1]

        # initialize clf_dict dict with clf as key
        # example:
        # clf_dict['RF'] return a list of Random Forest classifiers trained on each label
        # clf_dict['RF'][0] return a Random Forest trained on label nr 0

        clfs_dict = {}
        for clf in self.clfs:
            clfs_dict[clf] = []

        # train classifiers
        for clf in self.clfs:
            for k in range(0, n_labels):
                # print(k)
                # if logger is not None:  # used in offline eval only
                #     logger.log({f"k": k})
                if clf == "LR":
                    clfs_dict[clf].append(self.train_clf_lr(encodings, labels[:, k]))
                if clf == "RF":
                    clfs_dict[clf].append(
                        RandomForestClassifier(
                            n_estimators=self.f_n_estimators,
                            min_samples_split=self.f_min_samples_split,
                            min_samples_leaf=self.f_min_samples_leaf,
                            max_features=self.f_max_features,
                            max_depth=self.f_max_depth,
                            criterion=self.f_criterion,
                            bootstrap=self.f_bootstrap,
                            n_jobs=self.f_n_jobs,
                        ).fit(encodings, labels[:, k])
                    )
        return clfs_dict

    def get_training_output(self, training_out):
        if len(training_out) == 0:
            return
        enc_out_emb_train = {}
        if self.vision_modality_config is not None:
            enc_out_emb_train["vision"] = []
        if self.text_modality_config is not None:
            enc_out_emb_train["text"] = []

        labels_train = []
        # select samples for training of classifier
        for idx, train_out in enumerate(training_out):
            enc_embs, labels = train_out

            if "vision" in enc_embs.keys():
                # emb_m = enc_embs["vision"]["features"]
                emb_m = enc_embs["vision"]
                emb_m = emb_m.movedim(0, 1)
                enc_out_embs_m = enc_out_emb_train["vision"]
                # use CLS token for downstream task prediction
                enc_out_embs_m.append(emb_m[:, 0, :])
                enc_out_emb_train["vision"] = enc_out_embs_m
            if "text" in enc_embs.keys():
                emb_m = enc_embs["text"]
                enc_out_embs_m = enc_out_emb_train["text"]
                # use CLS token for downstream task prediction
                enc_out_embs_m.append(emb_m[:, 0, :])
                enc_out_emb_train["text"] = enc_out_embs_m
            labels_train.append(labels)

        masks = []
        for key in enc_out_emb_train.keys():
            enc_out_emb_m_train = enc_out_emb_train[key]
            enc_out_emb_m_train = torch.cat(enc_out_emb_m_train, dim=0)
            enc_out_emb_train[key] = enc_out_emb_m_train
            mask_m = ~torch.any(enc_out_emb_m_train.isnan(), dim=1)
            masks.append(mask_m.unsqueeze(1))
        masks = torch.any(torch.cat(masks, dim=1), dim=1)
        for key in enc_out_emb_train.keys():
            enc_m_train = enc_out_emb_train[key]
            enc_m_train_filtered = enc_m_train[masks]
            enc_out_emb_train[key] = enc_m_train_filtered
        labels_train = torch.cat(labels_train, dim=0)[masks]
        return enc_out_emb_train, labels_train

    def get_validation_output(self, validation_out):
        enc_emb_val = {}
        if self.vision_modality_config is not None:
            enc_emb_val["vision"] = []
        if self.text_modality_config is not None:
            enc_emb_val["text"] = []
        labels_val = []
        losses = []
        for idx, val_out in enumerate(validation_out):
            enc_embs, labels = val_out
            if "vision" in enc_embs.keys():
                # emb_m = enc_embs["vision"]["features"]
                emb_m = enc_embs["vision"]
                emb_m = emb_m.movedim(0, 1)
                enc_emb_m = enc_emb_val["vision"]
                # use CLS token for downstream task prediction
                enc_emb_m.append(emb_m[:, 0, :])
                enc_emb_val["vision"] = enc_emb_m
            if "text" in enc_embs.keys():
                emb_m = enc_embs["text"]
                enc_emb_m = enc_emb_val["text"]
                # use CLS token for downstream task prediction
                enc_emb_m.append(emb_m[:, 0, :])
                enc_emb_val["text"] = enc_emb_m
            labels_val.append(labels)
        masks = []
        for key in enc_emb_val.keys():
            enc_emb_m_val = enc_emb_val[key]
            enc_emb_m_val = torch.cat(enc_emb_m_val, dim=0)
            enc_emb_val[key] = enc_emb_m_val
            mask_m = ~torch.any(enc_emb_m_val.isnan(), dim=1)
            masks.append(mask_m.unsqueeze(1))
        masks = torch.any(torch.cat(masks, dim=1), dim=1)
        for key in enc_emb_val.keys():
            enc_emb_m_val = enc_emb_val[key]
            enc_m_val_filtered = enc_emb_m_val[masks]
            enc_emb_val[key] = enc_m_val_filtered
        labels_val = torch.cat(labels_val, dim=0)[masks]
        return enc_emb_val, labels_val

    def eval_lr(
        self,
        emb_train,
        labels_train,
        emb_val,
        labels_val,
        device,
    ):
        clfs = {}
        # train linear clfs on representations that are fed into decoder
        for key in emb_train.keys():
            emb_m_train = emb_train[key]
            clf_out_m = self.train_clfs(
                emb_m_train,
                labels_train,
            )
            clfs[key] = clf_out_m

        # validate the linear classifiers
        # on for m, key in enumerate(self.modality_names):
        logs = {}
        for key in emb_val.keys():
            clf_m = clfs[key]
            emb_m_val = emb_val[key]
            scores_m = self.eval_clfs(
                clf_m,
                emb_m_val,
                labels_val,
                device,
            )
            # scores_m : RF -> AUROC -> Label
            for clf_type in self.clfs:
                for metric in self.metrics:
                    logs[f"downstream/{key}/{clf_type}/{metric}"] = scores_m[clf_type][
                        metric
                    ].mean()
        return logs

    def plot_samples(self, batch, model):
        n_samples_plot = 8
        out = model(batch)
        mods_rec, masks, _, _, _ = out
        img_logs = {}
        try:
            if "vision" in batch[0].keys():
                vis_batch = batch[0]["vision"]
                v_key = list(vis_batch.keys())[0]
                view_samples = vis_batch[v_key]["data"]
                available = view_samples.shape[0]
                # print(f"samples for modality {key} to plot: Available: {available}, required: {n_samples_plot}")
                mod_m = view_samples[:n_samples_plot]
                mod_rec_m = mods_rec[f"vision_{v_key}"][:n_samples_plot]
                mask_m = masks[f"vision_{v_key}"][:n_samples_plot]
                predicted_val_img = mod_rec_m * mask_m + mod_m * (1 - mask_m)
                img = torch.cat([mod_m * (1 - mask_m), predicted_val_img, mod_m], dim=0)
                img = (img - img.min()) / (img.max() - img.min())
                # Generate report for images if vision to text is available
                if (
                    f"vision_{v_key}_to_text_inputs" in mods_rec
                    or "vision_to_text_inputs" in mods_rec
                ):
                    if "vision_to_text_inputs" in mods_rec:
                        vision_to_text_input = mods_rec["vision_to_text_inputs"]
                        gts = mods_rec["vision_to_text_gts"]
                        vision_to_text_input["backward_indexes"] = [
                            x[:, :n_samples_plot]
                            for x in vision_to_text_input["backward_indexes"]
                        ]
                        vision_to_text_input["latent_tokens"] = [
                            x[:, :n_samples_plot]
                            for x in vision_to_text_input["latent_tokens"]
                        ]
                    else:
                        vision_to_text_input = mods_rec[
                            f"vision_{v_key}_to_text_inputs"
                        ]
                        gts = mods_rec[f"vision_{v_key}_to_text_gts"]
                        vision_to_text_input["backward_indexes"] = vision_to_text_input[
                            "backward_indexes"
                        ][:, :n_samples_plot]
                        vision_to_text_input["latent_tokens"] = vision_to_text_input[
                            "latent_tokens"
                        ][:, :n_samples_plot]
                    vision_to_text_input["labels"] = vision_to_text_input["labels"][
                        :n_samples_plot
                    ]
                    gen_text = model.text_decoder.generate(vision_to_text_input)
                    gen_text = [
                        f"PRED: {gen_text[i]}\n GT: {gts[i]}"
                        for i in range(n_samples_plot)
                    ]
                    img = torch.cat(
                        [
                            img[i * n_samples_plot : (i + 1) * n_samples_plot]
                            for i in range(img.shape[0] // n_samples_plot)
                        ],
                        dim=-1,
                    )
                    img_logs["vision"] = (img, gen_text)
                else:
                    img = rearrange(
                        img, "(v h1 w1) c h w -> c (h1 h) (w1 v w)", w1=2, v=3
                    )
                    img_logs["vision"] = img
        except:
            print("ERROR IN THE PLOTTING - skipped")

        return img_logs

    def eval(self, model):
        training_out = model.training_encodings
        eval_out_dict = (
            model.validation_step_outputs
        )  # dictionary with dataset names as keys
        logs = {}
        self.auroc_per_dataset = []

        if len(training_out) > 0:
            emb_train, labels_train = self.get_training_output(training_out)
            scalar_logs = {}
            first_validation_batch_dict = model.first_val_batch_dict
            individual_dataset_aurocs = []  # Track individual datasets separately
            for d_name, eval_out in eval_out_dict.items():
                emb_val, labels_val = self.get_validation_output(eval_out)

                print(f"Evaluating dataset {d_name} at epoch {model.current_epoch + 1}")
                print(
                    "Total number of training encodings:",
                    len(labels_train),
                    len(emb_train),
                )
                print(
                    "Total number of validation encodings:",
                    len(labels_val),
                    len(emb_val),
                )

                if (
                    self.evaluate_lr
                    and (model.current_epoch + 1) % self.logging_frequency_lr == 0
                ):
                    logs_lr = self.eval_lr(
                        emb_train, labels_train, emb_val, labels_val, model.device
                    )
                    logs_lr = {
                        f"online_eval/{d_name}/{k}": v for k, v in logs_lr.items()
                    }
                    print(logs_lr)
                    scalar_logs.update(logs_lr)
                    # Only include individual datasets in the average (exclude joint datasets)
                    if not d_name.startswith("joint_"):
                        individual_dataset_aurocs.append(self.auroc_per_dataset[-1])

            print(self.auroc_per_dataset)
            # compute avg metrics only for individual datasets (excluding joint datasets)
            if len(individual_dataset_aurocs) > 1:
                avg_auroc = torch.mean(torch.stack(individual_dataset_aurocs))
                print(f"Average AUROC across individual datasets: {avg_auroc}")
                scalar_logs["online_eval/avg_AUROC"] = avg_auroc
            logs["scalars"] = scalar_logs

            logs["imgs"] = {}
            if (model.current_epoch + 1) % self.logging_frequency_plots == 0:
                for d_name, _ in eval_out_dict.items():
                    print(
                        f"Plotting samples for dataset {d_name} at epoch {model.current_epoch + 1}"
                    )
                    img_logs = self.plot_samples(
                        first_validation_batch_dict[d_name], model
                    )
                    # append the dataset name to the keys
                    img_logs = {f"imgs/{d_name}/{k}": v for k, v in img_logs.items()}
                    logs["imgs"].update(img_logs)

        model.validation_step_outputs.clear()
        return logs
