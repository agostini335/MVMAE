import os

# Set the environment variable
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from einops import rearrange

import wandb
from config.MyMVWSLConfig import MyMVWSLConfig
from utils.config_setup import get_decoders, get_encoders, load_config
from utils.generic import (
    EpochPrefixCallback,
    WandbCheckpointCallback,
    check_cfg_logic,
    get_data_loaders,
)

load_config()


@hydra.main(config_path="./config", config_name="config_base", version_base="1.1")
def run_experiment(cfg: MyMVWSLConfig):
    # Resolve and check the config
    print(OmegaConf.to_yaml(cfg, resolve=True))
    cfg = OmegaConf.structured(cfg)
    check_cfg_logic(cfg)

    # Set up the Wandb instance
    if cfg.log.wandb_local_instance:
        wandb.login(host=os.getenv("WANDB_LOCAL_URL"))
    elif not cfg.log.wandb_offline:
        wandb.login(host="https://api.wandb.ai")
    pl.seed_everything(cfg.training.seed, workers=True)

    core_cfg = HydraConfig.get()
    print("Hydra run dir:", core_cfg.run.dir)

    if cfg.tokenizermodule is not None:
        tokenizer = instantiate(cfg.tokenizermodule)
    else:
        tokenizer = None

    tf_train = instantiate(cfg.transformmodule_train)
    tf_eval = instantiate(cfg.transformmodule_eval)

    train_dst = instantiate(
        cfg.datamodule_train,
        text_modality_config=cfg.text_modality,
        vision_modality_config=cfg.vision_modality,
        transform=tf_train,
        tokenizer=tokenizer,
        vision_text_reconstruction=cfg.vision_to_text_decoder,
    )

    val_dst_list = []
    # Create individual validation datasets (always done - current behavior)
    for val_dataset_name in cfg.eval.validation_dataset_names:
        val_dst_list.append(
            instantiate(
                cfg.datamodule_eval,
                transform=tf_eval,
                vision_modality_config=cfg.vision_modality,
                text_modality_config=cfg.text_modality,
                tokenizer=tokenizer,
                vision_text_reconstruction=cfg.vision_to_text_decoder,
                selected_datasets=[val_dataset_name],
            )
        )

    validation_dataset_names = list(
        cfg.eval.validation_dataset_names
    )  # Individual datasets

    # Optionally add a joint validation dataset containing all validation datasets
    if cfg.eval.add_joint_validation_dataset:
        val_dst_list.append(
            instantiate(
                cfg.datamodule_eval,
                transform=tf_eval,
                vision_modality_config=cfg.vision_modality,
                text_modality_config=cfg.text_modality,
                tokenizer=tokenizer,
                vision_text_reconstruction=cfg.vision_to_text_decoder,
                selected_datasets=cfg.eval.validation_dataset_names,
            )
        )
        # Add the joint validation name to the list
        validation_dataset_names.append("joint_val_dataset")

    train_loader, val_loader_list = get_data_loaders(
        text_modality_config=cfg.text_modality,
        train_dst=train_dst,
        val_dst_list=val_dst_list,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.dataset.num_workers,
        vision_to_text_reconstruction=cfg.vision_to_text_decoder,
        tokenizer_module=cfg.tokenizermodule,
    )

    # Build model components
    encoders = get_encoders(
        cfg.encodermodule,
        cfg.textencodermodule,
        cfg.vision_modality,
        cfg.text_modality,
        cfg.model.compute_rec,
    )
    decoders = get_decoders(
        cfg.decodermodule,
        cfg.textdecodermodule,
        cfg.vision_modality,
        cfg.text_modality,
    )
    evaluator = instantiate(
        cfg.evalmodule,
        vision_modality_config=cfg.vision_modality,
        text_modality_config=cfg.text_modality,
    )

    # Init model
    vtt_decoder = cfg.vision_to_text_decoder
    emb_dim = encoders[list(encoders.keys())[0]].emb_dim
    vtt_decoder_module = instantiate(cfg.vision_to_text_decoder_module, emb_dim)
    model = instantiate(
        cfg.modelmodule,
        encoders=encoders,
        decoders=decoders,
        evaluator=evaluator,
        vision_to_text_decoder=vtt_decoder,
        vision_to_text_decoder_module=vtt_decoder_module,
        vision_modality_config=cfg.vision_modality,
        text_modality_config=cfg.text_modality,
        validation_dataset_names=validation_dataset_names,
    )
    assert model is not None
    summary = ModelSummary(model, max_depth=2)
    print(summary)

    # Convert both configs to dictionaries before merging
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    hydra_cfg = OmegaConf.to_container(core_cfg, resolve=True, throw_on_missing=False)
    wandb_cfg["hydra"] = hydra_cfg

    # setup logger and callbacks
    wandb_logger = WandbLogger(
        config=wandb_cfg,
        project=cfg.log.wandb_project_name,
        group=cfg.log.wandb_group,
        offline=cfg.log.wandb_offline,
        entity=cfg.log.wandb_entity,
        id=cfg.training.resume_run_id,
        resume="allow",
    )
    checkpoint_callback = WandbCheckpointCallback(
        wandb_logger=wandb_logger,
        monitor=cfg.log.wandb_checkpoint_metric,
        mode=cfg.log.wandb_checkpoint_mode,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    pretrain_epoch_cb = EpochPrefixCallback("pretrain")
    offline_eval_epoch_cb = EpochPrefixCallback("offline_eval")

    # Pre-training Phase
    if cfg.training.do_pretraining:
        print("Starting pre-training")
        trainer = pl.Trainer(
            max_epochs=cfg.training.n_epochs,
            devices=cfg.training.num_gpus,
            log_every_n_steps=cfg.log.wandb_log_freq,
            num_nodes=cfg.training.num_nodes,
            accelerator="gpu",
            strategy=cfg.training.strategy,
            logger=wandb_logger,
            check_val_every_n_epoch=cfg.eval.validation_every_n_epochs,
            gradient_clip_val=cfg.training.gradient_clipping,
            deterministic=True,
            callbacks=[lr_monitor, checkpoint_callback, pretrain_epoch_cb],
        )

        # create training data loader for easier, thread-safe access in the online eval phase
        train_loader_rank_zero, _ = get_data_loaders(
            text_modality_config=cfg.text_modality,
            train_dst=train_dst,
            val_dst_list=val_dst_list,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.dataset.num_workers,
            vision_to_text_reconstruction=cfg.vision_to_text_decoder,
            tokenizer_module=cfg.tokenizermodule,
        )

        model.assign_rank_zero_train_dl(train_loader_rank_zero)

        if cfg.log.debug:
            trainer.logger.watch(model, log="all")

        # Check if we should resume from checkpoint based on resume_run_id
        ckpt_path = None
        if cfg.training.resume_run_id:
            run_path = os.path.join(hydra_cfg["run"]["dir"], "wandb")
            # search for  the latest modified folder in run_path that ends with the string "-" + resume_run_id, if run_path exists
            if os.path.exists(run_path):
                latest_time = 0
                for folder in os.listdir(run_path):
                    if folder.endswith(f"-{cfg.training.resume_run_id}"):
                        folder_path = os.path.join(run_path, folder)
                        mod_time = os.path.getmtime(folder_path)
                        if mod_time > latest_time:
                            latest_time = mod_time
                            ckpt_path = os.path.join(
                                folder_path, "files", "checkpoints", "last.ckpt"
                            )
            else:
                raise ValueError(
                    f"Run path {run_path} does not exist. Cannot resume from run id {cfg.training.resume_run_id}."
                )
            print(f"Resuming from checkpoint: {ckpt_path}")

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader_list,
            ckpt_path=ckpt_path,
        )

    # Offline Eval Phase
    if cfg.offline_eval.do_offline_eval:
        pl.seed_everything(cfg.offline_eval.seed, workers=True)
        print("Starting offline evaluation")
        if cfg.training.do_pretraining:
            print("Using in-memory encoders")
        elif cfg.offline_eval.checkpoint_init:
            print("Loading encoders from checkpoint.")
            assert os.path.exists(
                cfg.offline_eval.ckpt_path
            ), f"Checkpoint not found: {cfg.offline_eval.ckpt_path}"
            state_dict = torch.load(cfg.offline_eval.ckpt_path, weights_only=False)[
                "state_dict"
            ]
            model.load_state_dict(state_dict, strict=False)
        else:
            print("Using randomly initialized encoders for supervised baseline.")
        la_module = None
        clf = instantiate(
            cfg.offline_evalmodule,
            oe_encoders=model.encoders,
            oe_validation_dataset_names=validation_dataset_names,
            oe_latent_attention_module=la_module,
        )
        assert clf
        print("Classifier Summary")
        print(ModelSummary(clf, max_depth=2))

        # Train the classifier
        trainer = pl.Trainer(
            max_epochs=cfg.offline_evalmodule.oe_n_epochs,
            devices=cfg.offline_evalmodule.oe_num_gpus,
            log_every_n_steps=cfg.log.wandb_log_freq,
            num_nodes=cfg.offline_evalmodule.oe_num_nodes,
            accelerator="gpu",
            strategy=cfg.training.strategy,
            logger=wandb_logger,
            deterministic=True,
            callbacks=[lr_monitor, offline_eval_epoch_cb],
        )
        trainer.fit(
            clf, train_dataloaders=train_loader, val_dataloaders=val_loader_list
        )

    wandb.finish()


if __name__ == "__main__":
    load_config()
    run_experiment()
