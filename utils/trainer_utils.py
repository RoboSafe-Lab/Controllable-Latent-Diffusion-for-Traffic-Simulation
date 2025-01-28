from tbsim.utils.config_utils import translate_pass_trajdata_cfg
from tbsim.datasets.trajdata_datamodules import PassUnifiedDataModule
from datetime import  datetime
import os,json,wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from configs.custom_config import serialize_object
from trainers.vae_trainer import VAELightningModule
from trainers.dm_trainer import DMLightningModule
from trainers.guide_dm_trainer import GuideDMLightningModule
from configs.visualize_traj import TrajectoryVisualizationCallback
def prepare_trainer_and_data(cfg, train_mode,debug=False):
    trajdata_config = translate_pass_trajdata_cfg(cfg)
    datamodule = PassUnifiedDataModule(trajdata_config, cfg.train)
    datamodule.setup()

    checkpoint_vae = cfg.train.checkpoint_vae
    checkpoint_dm = cfg.train.checkpoint_dm
    if train_mode == "vae":
        model = VAELightningModule( 
            algo_config=cfg.algo,
            train_config=cfg.train,
            modality_shapes=datamodule.modality_shapes,
                                    )
    elif train_mode == "dm":
        model = DMLightningModule(
            algo_config=cfg.algo,
            train_config=cfg.train,
            modality_shapes=datamodule.modality_shapes,
            vae_model_path = checkpoint_vae,
                           )
    else:
         raise ValueError(f"Unknown train mode: {train_mode}")
    
    train_callbacks = []
    logger= None

    if not debug:
        wandb_base, time= create_wandb_dir(base_dir=cfg.train.wandb_dir)
        config_path = os.path.join(wandb_base, "config.json")
        with open(config_path, "w") as f:
            json.dump(serialize_object(cfg), f, indent=4)
        wandb.login()
        logger = WandbLogger(
            name=f"{cfg.name}_{time}",
            project=cfg.train.logging.wandb_project_name,
            save_dir='wandb_cache'
            )
        logger.watch(model)
        logger.experiment.config.update(cfg.to_dict())
    
        checkpoint_dir, media_dir = [os.path.join(wandb_base, subdir) for subdir in ("checkpoints", "media")]

        if cfg.train.validation.enabled:##NOTE:  first validation then save
            assert (cfg.train.save.every_n_steps > cfg.train.validation.every_n_steps),"checkpointing frequency (" + str(
                cfg.train.save.every_n_steps) + ") needs to be greater than validation frequency (" + str(cfg.train.validation.every_n_steps) + ")"
            
            ckpt_valid_callback = pl.callbacks.ModelCheckpoint(
                dirpath=f"{checkpoint_dir}",
                filename=f"iter{{step}}_ep{{epoch}}_val/loss",
                auto_insert_metric_name=False,
                save_top_k=cfg.train.save.best_k,
                monitor='val/loss',
                mode="min",
                every_n_train_steps=cfg.train.save.every_n_steps,
                verbose=True,
                
            )
            train_callbacks.append(ckpt_valid_callback)
        visual_callback = TrajectoryVisualizationCallback(cfg,media_dir)
        train_callbacks.append(visual_callback)
    else:
        
        checkpoint_dir = "logs/debug_run"
        os.makedirs(checkpoint_dir, exist_ok=True)
        media_dir = os.path.join(checkpoint_dir, "media")
        os.makedirs(media_dir, exist_ok=True)

        print("Debug mode: skipping checkpoint callbacks")
        vis_callback = TrajectoryVisualizationCallback(cfg, media_dir)
        train_callbacks.append(vis_callback)

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
    return trainer,datamodule, model,checkpoint_vae,checkpoint_dm
            
def create_wandb_dir(base_dir="logs"):
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, timestamp


def prepare_for_guided_dm(cfg,debug=False):
    trajdata_config = translate_pass_trajdata_cfg(cfg)
    datamodule = PassUnifiedDataModule(trajdata_config, cfg.train)
    datamodule.setup()

    
    checkpoint_dm = cfg.train.checkpoint_dm
    checkpoint_rl = cfg.train.checkpoint_rl
    model = GuideDMLightningModule(
        algo_config=cfg.algo,
        train_config=cfg.train,
        modality_shapes=datamodule.modality_shapes,
        dm_model_path=checkpoint_dm,
        rl_model_path= checkpoint_rl,
                    )

    
    train_callbacks = []
    logger= None

    if not debug:
        wandb_base, time= create_wandb_dir(base_dir=cfg.train.wandb_dir)
        config_path = os.path.join(wandb_base, "config.json")
        with open(config_path, "w") as f:
            json.dump(serialize_object(cfg), f, indent=4)
        wandb.login()
        logger = WandbLogger(
            name=f"{cfg.name}_{time}",
            project=cfg.train.logging.wandb_project_name,
            save_dir='wandb_cache'
            )
        logger.watch(model)
        logger.experiment.config.update(cfg.to_dict())
    
        checkpoint_dir, media_dir = [os.path.join(wandb_base, subdir) for subdir in ("checkpoints", "media")]

        if cfg.train.validation.enabled:##NOTE:  first validation then save
            assert (cfg.train.save.every_n_steps > cfg.train.validation.every_n_steps),"checkpointing frequency (" + str(
                cfg.train.save.every_n_steps) + ") needs to be greater than validation frequency (" + str(cfg.train.validation.every_n_steps) + ")"
            
            ckpt_valid_callback = pl.callbacks.ModelCheckpoint(
                dirpath=f"{checkpoint_dir}",
                filename=f"iter{{step}}_ep{{epoch}}_val/loss",
                auto_insert_metric_name=False,
                save_top_k=cfg.train.save.best_k,
                monitor='val/loss',
                mode="min",
                every_n_train_steps=cfg.train.save.every_n_steps,
                verbose=True,
                
            )
            train_callbacks.append(ckpt_valid_callback)
        visual_callback = TrajectoryVisualizationCallback(cfg,media_dir)
        train_callbacks.append(visual_callback)
    else:
        
        checkpoint_dir = "logs/debug_run"
        os.makedirs(checkpoint_dir, exist_ok=True)
        media_dir = os.path.join(checkpoint_dir, "media")
        os.makedirs(media_dir, exist_ok=True)

        print("Debug mode: skipping checkpoint callbacks")
        vis_callback = TrajectoryVisualizationCallback(cfg, media_dir)
        train_callbacks.append(vis_callback)

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
    
            
    return trainer,datamodule,model,ckpt