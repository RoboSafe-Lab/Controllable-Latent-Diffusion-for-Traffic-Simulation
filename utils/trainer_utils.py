from tbsim.utils.config_utils import translate_pass_trajdata_cfg
from datetime import  datetime
import os,json,wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from configs.custom_config import serialize_object
from trainers.vae_trainer import VAELightningModule
from trainers.dm_trainer import DMLightningModule
from trainers.guide_dm_trainer import GuideDMLightningModule
from configs.visualize_traj import TrajectoryVisualizationCallback,VisierProgressBar
from configs.datamodules import Hf_DataModule
import torch

def prepare_trainer_and_data(cfg, train_mode,debug=False):
    trajdata_config = translate_pass_trajdata_cfg(cfg)
    datamodule = Hf_DataModule(trajdata_config, cfg.train)
    datamodule.setup()

    ckpt_vae = cfg.train.checkpoint_vae
    ckpt_dm = cfg.train.checkpoint_dm
    
    if train_mode == "dm":
        model = DMLightningModule(
            algo_config=cfg.algo,
            train_config=cfg.train,
            modality_shapes=datamodule.modality_shapes,
            ckpt_dm=ckpt_dm,
                           )
        if ckpt_vae is not None and os.path.exists(ckpt_vae):
            print(f"Loading VAE weights from {ckpt_vae}")
            vae_ckpt = torch.load(ckpt_vae,map_location='cpu')
            vae_state = {}
            prefix = 'vae.'

            for old_key, value in vae_ckpt['state_dict'].items():
                if old_key.startswith(prefix):
                    new_key = old_key[len(prefix):]
                    vae_state[new_key]=value
           
            missing, unexpected = model.vae.load_state_dict(vae_state, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
    elif train_mode == 'vae':
        model = VAELightningModule(
            algo_config=cfg.algo,
            train_config=cfg.train,
            modality_shapes=datamodule.modality_shapes,
        )
    
    train_callbacks = []
    logger= None

    if not debug:
        wandb_base, time= create_wandb_dir(train_mode,cfg.train.wandb_dir)
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
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(media_dir, exist_ok=True)
        if cfg.train.validation.enabled:##NOTE:  first validation then save
            assert (cfg.train.save.every_n_steps > cfg.train.validation.every_n_steps),"checkpointing frequency (" + str(
                cfg.train.save.every_n_steps) + ") needs to be greater than validation frequency (" + str(cfg.train.validation.every_n_steps) + ")"
            
            #
            ckpt_valid_callback = pl.callbacks.ModelCheckpoint(
                monitor = 'val/loss',
                mode="min",
                save_top_k = 1,
                dirpath=f"{checkpoint_dir}",
                filename=f"iter{{step}}_ep{{epoch}}---val/loss",
                save_on_train_epoch_end = True,
                # every_n_train_steps= None, #cfg.train.save.every_n_steps,
                verbose=True,  
            )
            train_callbacks.append(ckpt_valid_callback)
        visual_callback = TrajectoryVisualizationCallback(cfg,media_dir)
        train_callbacks.append(visual_callback)
        train_callbacks.append(VisierProgressBar())
    else:
        
        checkpoint_dir = "logs/debug_run"
        os.makedirs(checkpoint_dir, exist_ok=True)
        media_dir = os.path.join(checkpoint_dir, "media")
        os.makedirs(media_dir, exist_ok=True)

        print("Debug mode: skipping checkpoint callbacks")
        vis_callback = TrajectoryVisualizationCallback(cfg, media_dir)
        train_callbacks.append(vis_callback)

    trainer = pl.Trainer(
    precision='16-mixed',
    default_root_dir=checkpoint_dir,
    # checkpointing
    enable_checkpointing=cfg.train.save.enabled,
    # logging
    logger=logger,
    # flush_logs_every_n_steps=cfg.train.logging.flush_every_n_steps,
    log_every_n_steps=cfg.train.logging.log_every_n_steps,
    # training
    max_epochs = cfg.train.training.epochs,
    # max_steps=cfg.train.training.num_steps,
    # validation
    check_val_every_n_epoch=1,
    val_check_interval=None,#cfg.train.validation.every_n_steps,
    limit_val_batches=cfg.train.validation.num_steps_per_epoch,
    # all callbacks
    callbacks=train_callbacks,
    num_sanity_val_steps=0,
    gradient_clip_val=1.0,
    
    
)   
    if train_mode=='vae':
        return trainer, datamodule, model, ckpt_vae
    else:
        return trainer, datamodule, model
            
def create_wandb_dir(train_mode,base_dir="logs"):
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder_name = f"{timestamp} {train_mode}"
    run_dir = os.path.join(base_dir, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, timestamp


def prepare_for_guided_dm(cfg,debug=False):
    trajdata_config = translate_pass_trajdata_cfg(cfg)
    datamodule = Hf_DataModule(trajdata_config, cfg.train)
    datamodule.setup()

    ckpt_vae = cfg.train.checkpoint_vae
    ckpt_dm = cfg.train.checkpoint_dm

    model = GuideDMLightningModule(
        algo_config=cfg.algo,
        train_config=cfg.train,
        modality_shapes=datamodule.modality_shapes,
        ckpt_dm = cfg.train.checkpoint_dm
      
                    )
    if ckpt_vae is not None and os.path.exists(ckpt_vae):
            print(f"Loading VAE weights from {ckpt_vae}")
            vae_ckpt = torch.load(ckpt_vae,map_location='cpu')
            vae_state = {}
            prefix = 'vae.'

            for old_key, value in vae_ckpt['state_dict'].items():
                if old_key.startswith(prefix):
                    new_key = old_key[len(prefix):]
                    vae_state[new_key]=value
           
            missing, unexpected = model.vae.load_state_dict(vae_state, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

    
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
                monitor = 'val/loss',
                mode="min",
                save_top_k = 1,
                dirpath=f"{checkpoint_dir}",
                filename=f"iter{{step}}_ep{{epoch}}---val/loss",
                save_on_train_epoch_end = True,
                # every_n_train_steps= None, #cfg.train.save.every_n_steps,
                verbose=True,  
            )
            train_callbacks.append(ckpt_valid_callback)
        visual_callback = TrajectoryVisualizationCallback(cfg,media_dir)
        train_callbacks.append(visual_callback)
        train_callbacks.append(VisierProgressBar())
    else:
        
        checkpoint_dir = "logs/debug_run"
        os.makedirs(checkpoint_dir, exist_ok=True)
        media_dir = os.path.join(checkpoint_dir, "media")
        os.makedirs(media_dir, exist_ok=True)

        print("Debug mode: skipping checkpoint callbacks")
        vis_callback = TrajectoryVisualizationCallback(cfg, media_dir)
        train_callbacks.append(vis_callback)

    trainer = pl.Trainer(
    precision='16-mixed',
    default_root_dir=checkpoint_dir,
    # checkpointing
    enable_checkpointing=cfg.train.save.enabled,
    # logging
    logger=logger,
    # flush_logs_every_n_steps=cfg.train.logging.flush_every_n_steps,
    log_every_n_steps=cfg.train.logging.log_every_n_steps,
    # training
    max_epochs = cfg.train.training.epochs,
    # max_steps=cfg.train.training.num_steps,
    # validation
    check_val_every_n_epoch=1,
    val_check_interval=None,#cfg.train.validation.every_n_steps,
    limit_val_batches=cfg.train.validation.num_steps_per_epoch,
    # all callbacks
    callbacks=train_callbacks,
    num_sanity_val_steps=0,
    gradient_clip_val=1.0,
    
    
) 
    
            
    return trainer,datamodule,model