import os
import argparse
import  sys
import pytorch_lightning as pl

from tbsim.utils.log_utils import PrintLogger
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
import tbsim.utils.train_utils as TrainUtils
from tbsim.datasets.factory import datamodule_factory
from tbsim.utils.env_utils import RolloutCallback

import wandb,json
from pytorch_lightning.loggers import  WandbLogger
from  models.algos import  UnifiedTrainer
from datetime import  datetime
from configs.custom_config import dict_to_config,ConfigBase,serialize_object
from tbsim.configs.base import ExperimentConfig
import yaml,torch



def create_wandb_dir(base_dir="logs"):
    """
    Create a directory under the wandb base directory with a timestamp as the name.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main(cfg, debug=False):
    pl.seed_everything(cfg.seed)
    set_global_batch_type("trajdata")
    set_global_trajdata_batch_env(cfg.train.trajdata_source_train[0])
    set_global_trajdata_batch_raster_cfg(cfg.env.rasterizer)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n============= New Training Run with Config =============")

    


    datamodule = datamodule_factory(cls_name=cfg.train.datamodule_class, config=cfg)
    datamodule.setup()
    
    checkpoint_vae = cfg.train.checkpoint_vae
    checkpoint_dm = cfg.train.checkpoint_dm
    model = UnifiedTrainer(algo_config=cfg.algo,train_config=cfg.train,
                           modality_shapes=datamodule.modality_shapes,
                           train_mode=cfg.train.mode,
                           vae_model_path = checkpoint_vae,
                           )


    logger = None
    train_callbacks = []
    if not debug:
        wandb_base_dir = "logs"
        wandb_run_dir = create_wandb_dir(base_dir=wandb_base_dir)
        config_path = os.path.join(wandb_run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(serialize_object(cfg), f, indent=4)
        wandb.login()
        logger = WandbLogger(
            name=f"{cfg.name}_{current_time}",
            project=cfg.train.logging.wandb_project_name,
            save_dir=wandb_run_dir
            )
        logger.watch(model=model)
        logger.experiment.config.update(cfg.to_dict())

        checkpoint_dir = os.path.join(wandb_run_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
       

        
        if cfg.train.rollout.enabled:
        
            rollout_callback = RolloutCallback(
                exp_config=cfg,
                every_n_steps=cfg.train.rollout.every_n_steps,
                warm_start_n_steps=cfg.train.rollout.warm_start_n_steps,
                verbose=True,
                save_video=cfg.train.rollout.save_video,
                video_dir=os.path.join(wandb_run_dir, "videos")
            )
            train_callbacks.append(rollout_callback)
        
        # Checkpointing
        if cfg.train.validation.enabled and cfg.train.save.save_best_validation:#NOTE:  first validation then save
            assert (cfg.train.save.every_n_steps > cfg.train.validation.every_n_steps),"checkpointing frequency (" + str(
                cfg.train.save.every_n_steps) + ") needs to be greater than validation frequency (" + str(cfg.train.validation.every_n_steps) + ")"
            
            
            for metric_name, metric_key in model.checkpoint_monitor_keys.items():
                print(
                    "Monitoring metrics {} under alias {}".format(metric_key, metric_name)
                )
                ckpt_valid_callback = pl.callbacks.ModelCheckpoint(
                    dirpath=f"{checkpoint_dir}/{metric_name}",
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
                dirpath=checkpoint_dir,
                filename="iter{step}_ep{epoch}_simADE{rollout/metrics_ego_ADE:.2f}",
                auto_insert_metric_name=False,
                save_top_k=cfg.train.save.best_k,  # save the best k models
                monitor="rollout/metrics_ego_ADE",
                mode="min",
                every_n_train_steps=cfg.train.save.every_n_steps,
                state_key='rollout_checkpoint',
                verbose=True,
            )
            train_callbacks.append(ckpt_rollout_callback)


        images_dir = os.path.join(wandb_run_dir, "images")
        os.makedirs(images_dir, exist_ok=True)


        model.image_dir = images_dir
    else:
        wandb_run_dir = "logs/debug_run"
        os.makedirs(wandb_run_dir, exist_ok=True)
        checkpoint_dir = wandb_run_dir
        print("Debug mode: skipping checkpoint callbacks")
    trainer = pl.Trainer(
       
        default_root_dir=checkpoint_dir,
        # checkpointing
        enable_checkpointing=cfg.train.save.enabled,
        # logging
        logger=logger,
        # flush_logs_every_n_steps=cfg.train.logging.flush_every_n_steps,
        log_every_n_steps=cfg.train.logging.log_every_n_steps,
        # training
        min_epochs = 1,
        # max_steps=cfg.train.training.num_steps,
        # validation
        val_check_interval=cfg.train.validation.every_n_steps,
        limit_val_batches=cfg.train.validation.num_steps_per_epoch,
        # all callbacks
        callbacks=train_callbacks,
        num_sanity_val_steps=0,
        
       
    )
    # checkpoint_point  = "/home/visier/hazardforge/HazardForge/checkpoint/vae/loss.ckpt"
    # trainer.fit(model=model, datamodule=datamodule,ckpt_path=checkpoint_dm)
    trainer.test(model=model, datamodule=datamodule,ckpt_path=checkpoint_dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--config", type=str, default="./config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    train_config = dict_to_config(ConfigBase, config_dict.get("train", {}))
    env_config = dict_to_config(ConfigBase, config_dict.get("env", {}))
    algo_config = dict_to_config(ConfigBase, config_dict.get("algo", {}))
    default_config = ExperimentConfig(
        train_config=train_config,
        env_config=env_config,
        algo_config=algo_config,
        registered_name=config_dict.get("registered_name", "default_experiment"),
    )
  

    if default_config.train.rollout.get("enabled", False):
        default_config.env["eval"] = {"env": default_config.env["name"]}
        assert default_config.algo["eval_class"], f"Please set an eval_class for {default_config.algo['name']}"
        default_config.env["eval"]["eval_class"] = default_config.algo["eval_class"]
        default_config.env["eval"]["dataset_path"] = default_config.train["trajdata_data_dirs"]["nusc_trainval"]
        env_specific_config = default_config.env.get(default_config.env["eval"]["env"], {})
        for key, value in env_specific_config.items():
            default_config.env["eval"][key] = value

    default_config.lock()  # Make config read-only
  
    main(default_config, debug=default_config.train.debug)
