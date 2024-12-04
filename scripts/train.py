import os
import argparse
import  sys
import pytorch_lightning as pl
from tbsim.utils.log_utils import PrintLogger
from configs.my_config_registry import  EXP_CONFIG_REGISTRY
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
import tbsim.utils.train_utils as TrainUtils
from tbsim.datasets.factory import datamodule_factory
from my_config_registry import get_registered_experiment_config
from tbsim.utils.env_utils import RolloutCallback
from tbsim.algos.factory import algo_factory
import wandb
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from  models.algos import  UnifiedTrainer
from datetime import  datetime



def main(cfg, auto_remove_exp_dir=True, debug=False):
    pl.seed_everything(cfg.seed)

    set_global_batch_type("trajdata")
    set_global_trajdata_batch_env(cfg.train.trajdata_source_train[0])
    set_global_trajdata_batch_raster_cfg(cfg.env.rasterizer)
    print("\n============= New Training Run with Config =============")
    root_dir, log_dir, ckpt_dir, video_dir, version_key = TrainUtils.get_exp_dir(
        exp_name=cfg.name,
        output_dir=cfg.root_dir,
        save_checkpoints=cfg.train.save.enabled,
        auto_remove_exp_dir=True
    )
    cfg.dump(os.path.join(root_dir, version_key, "config.json"))
    if cfg.train.logging.terminal_output_to_txt and not debug:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_callbacks = []
    # Training Parallelism
    assert cfg.train.parallel_strategy in [
        "dp",
        "ddp_spawn",
        None,
    ]
    if not cfg.devices.num_gpus > 1:

        with cfg.train.unlocked():
            cfg.train.parallel_strategy = None
    if cfg.train.parallel_strategy in ["ddp_spawn"]:
        with cfg.train.training.unlocked():
            cfg.train.training.batch_size = int(
                cfg.train.training.batch_size / cfg.devices.num_gpus
            )
        with cfg.train.validation.unlocked():
            cfg.train.validation.batch_size = int(
                cfg.train.validation.batch_size / cfg.devices.num_gpus
            )

    datamodule = datamodule_factory(
        cls_name=cfg.train.datamodule_class, config=cfg
    )
    datamodule.setup()

    # Environment for close-loop evaluation
    if cfg.train.rollout.enabled:
        # Run rollout at regular intervals
        rollout_callback = RolloutCallback(
            exp_config=cfg,
            every_n_steps=cfg.train.rollout.every_n_steps,
            warm_start_n_steps=cfg.train.rollout.warm_start_n_steps,
            verbose=True,
            save_video=cfg.train.rollout.save_video,
            video_dir=video_dir
        )
        train_callbacks.append(rollout_callback)
    model = UnifiedTrainer(algo_config=cfg.algo,train_config=cfg.train,
                           modality_shapes=datamodule.modality_shapes,
                           registered_name=cfg.registered_name,
                           train_mode=args.train_mode)
    # Checkpointing
    if cfg.train.validation.enabled and cfg.train.save.save_best_validation:
        assert (
                cfg.train.save.every_n_steps > cfg.train.validation.every_n_steps
        ), "checkpointing frequency (" + str(
            cfg.train.save.every_n_steps) + ") needs to be greater than validation frequency (" + str(
            cfg.train.validation.every_n_steps) + ")"
        for metric_name, metric_key in model.checkpoint_monitor_keys.items():
            print(
                "Monitoring metrics {} under alias {}".format(metric_key, metric_name)
            )
            ckpt_valid_callback = pl.callbacks.ModelCheckpoint(
                dirpath=f"{ckpt_dir}/{metric_name}",
                filename=f"iter{{step}}_ep{{epoch}}_{metric_name}_{metric_key}",
                auto_insert_metric_name=False,
                save_top_k=cfg.train.save.best_k,
                monitor=metric_key,
                mode="min",
                every_n_train_steps=cfg.train.save.every_n_steps,
                verbose=True,
            )
            train_callbacks.append(ckpt_valid_callback)
    if cfg.train.rollout.enabled and cfg.train.save.save_best_rollout:
        assert (
            cfg.train.save.every_n_steps > cfg.train.rollout.every_n_steps
        ), "checkpointing frequency needs to be greater than rollout frequency"
        ckpt_rollout_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="iter{step}_ep{epoch}_simADE{rollout/metrics_ego_ADE:.2f}",
            # explicitly spell out metric names, otherwise PL parses '/' in metric names to directories
            auto_insert_metric_name=False,
            save_top_k=cfg.train.save.best_k,  # save the best k models
            monitor="rollout/metrics_ego_ADE",
            mode="min",
            every_n_train_steps=cfg.train.save.every_n_steps,
            state_key='rollout_checkpoint',

            verbose=True,
        )
        train_callbacks.append(ckpt_rollout_callback)


    # Logging
    assert not (cfg.train.logging.log_tb and cfg.train.logging.log_wandb)
    logger = None
    if debug:
        print("Debugging mode, suppress logging.")
    elif cfg.train.logging.log_tb:
        logger = TensorBoardLogger(
            save_dir=root_dir, version=version_key, name=None, sub_dir="logs/"
        )
        print("Tensorboard event will be saved at {}".format(logger.log_dir))
    elif cfg.train.logging.log_wandb:
        assert (
            "WANDB_APIKEY" in os.environ
        ), "Set api key by `export WANDB_APIKEY=<your-apikey>`"
        apikey = os.environ["WANDB_APIKEY"]
        wandb.login(key=apikey)
        logger = WandbLogger(
            name=f"{cfg.name}_{current_time}",
            project=cfg.train.logging.wandb_project_name
        )
        # record the entire config on wandb
        logger.experiment.config.update(cfg.to_dict())
        logger.watch(model=model)
    else:
        print("WARNING: not logging training stats")

    trainer = pl.Trainer(
        default_root_dir=root_dir,
        # checkpointing
        enable_checkpointing=cfg.train.save.enabled,
        # logging
        logger=logger,
        # flush_logs_every_n_steps=cfg.train.logging.flush_every_n_steps,
        log_every_n_steps=1,#cfg.train.logging.log_every_n_steps,
        # training
        max_steps=cfg.train.training.num_steps,
        # validation
        val_check_interval=cfg.train.validation.every_n_steps,
        limit_val_batches=cfg.train.validation.num_steps_per_epoch,
        # all callbacks
        callbacks=train_callbacks,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=model, datamodule=datamodule)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")



    parser.add_argument(
        "--config_name",
        type=str,
        default="nusc_hf",
        help="(optional) create experiment config from a preregistered name (see configs/my_config_registry.py)",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset root path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visier_logs/",
        help="Root directory of training output (checkpoints, visualization, tensorboard log, etc.)",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help="(optional) if provided, override the wandb project name defined in the config",
    )
    parser.add_argument(
        "--remove_exp_dir",
        action="store_true",
        help="Whether to automatically remove existing experiment directory of the same name (remember to set this to "
             "True to avoid unexpected stall when launching cloud experiments).",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode, suppress wandb logging, etc."
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        choices=["vae", "dm"],
        default="vae",
        help="Specify which model to train: 'vae' for Variational Autoencoder, 'dm' for Diffusion Model",
    )

    args = parser.parse_args()
    default_config = get_registered_experiment_config(args.config_name)
    print('args.config_name: ', args.config_name)
    print('default_config', default_config)

    if args.dataset_path is not None:
        default_config.train.dataset_path = args.dataset_path
        for key in default_config.eval.trajdata.trajdata_data_dirs:
            default_config.eval.trajdata.trajdata_data_dirs[key] = args.dataset_path

    if args.output_dir is not None:
        default_config.root_dir = os.path.abspath(args.output_dir)

    if args.wandb_project_name is not None:
        default_config.train.logging.wandb_project_name = args.wandb_project_name

    if args.debug:
        # Test policy rollout
        default_config.train.validation.every_n_steps = 5
        default_config.train.save.every_n_steps = 10
        default_config.train.rollout.every_n_steps = 10
        default_config.train.rollout.num_episodes = 1

    if default_config.train.rollout.enabled:
        default_config.eval.env = default_config.env.name
        assert default_config.algo.eval_class is not None, \
            "Please set an eval_class for {}".format(default_config.algo.name)
        default_config.eval.eval_class = default_config.algo.eval_class
        default_config.eval.dataset_path = default_config.train.dataset_path
        for k in default_config.eval[default_config.eval.env]:  # copy env-specific config to the global-level
            default_config.eval[k] = default_config.eval[default_config.eval.env][k]
        default_config.eval.pop("nusc")
        default_config.eval.pop("l5kit")

    default_config.lock()  # Make config read-only
    main(default_config, auto_remove_exp_dir=args.remove_exp_dir, debug=args.debug)
