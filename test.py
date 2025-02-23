import yaml,argparse
import pytorch_lightning as pl
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from configs.custom_config import dict_to_config,ConfigBase
from tbsim.configs.base import ExperimentConfig
from utils.trainer_utils import prepare_trainer_and_data


def train_vae(cfg,debug=False):
    trainer, datamodule,model,ckpt_vae= prepare_trainer_and_data(cfg,train_mode="vae",debug=cfg.train.debug)
    trainer.fit(model=model, datamodule=datamodule,ckpt_path=ckpt_vae)

def train_dm(cfg,debug=False):
    trainer, datamodule,model = prepare_trainer_and_data(cfg,train_mode="dm",debug=cfg.train.debug)  
    trainer.fit(model=model, datamodule=datamodule)

def train_ppo(cfg,debug=False):
    trainer, datamodule,model = prepare_trainer_and_data(cfg,train_mode='ppo',debug=cfg.train.debug)
    trainer.fit(model=model,datamodule=datamodule)
 
def main(cfg):
    pl.seed_everything(cfg.seed)
    set_global_batch_type("trajdata")
    set_global_trajdata_batch_env(cfg.train.trajdata_source_train[0])
    set_global_trajdata_batch_raster_cfg(cfg.env.rasterizer)
    
    print("\n============= New Training Run with Config =============")

    if cfg.train.mode == "vae":
        train_vae(cfg)
    elif cfg.train.mode == "dm":
        train_dm(cfg)
    elif cfg.train.mode == 'ppo':
        train_ppo(cfg)
    else:
        raise ValueError(f"Unknown train mode: {cfg.train.mode}") 

    
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
    default_config.lock()  # Make config read-only
    main(default_config)
