import os,yaml,argparse
import pytorch_lightning as pl
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from datetime import  datetime
from configs.custom_config import dict_to_config,ConfigBase
from tbsim.configs.base import ExperimentConfig
from utils.trainer_utils import prepare_trainer_and_data


def train_vae(cfg,debug=False):
    trainer, datamodule,model,ckpt_vae,_ = prepare_trainer_and_data(cfg,train_mode="vae",debug=cfg.train.debug)
    trainer.fit(model=model, datamodule=datamodule,ckpt_path=ckpt_vae)

def train_dm(cfg,debug=False):
    trainer, datamodule,model,_,ckpt_dm = prepare_trainer_and_data(cfg,train_mode="dm",debug=cfg.train.debug)
    trainer.fit(model=model, datamodule=datamodule,ckpt_path=ckpt_dm)


def create_wandb_dir(base_dir="logs"):
    """
    Create a directory under the wandb base directory with a timestamp as the name.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

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
  

    if default_config.train.rollout.get("enabled", False):
        default_config.env["eval"] = {"env": default_config.env["name"]}
        assert default_config.algo["eval_class"], f"Please set an eval_class for {default_config.algo['name']}"
        default_config.env["eval"]["eval_class"] = default_config.algo["eval_class"]
        default_config.env["eval"]["dataset_path"] = default_config.train["trajdata_data_dirs"]["nusc_trainval"]
        env_specific_config = default_config.env.get(default_config.env["eval"]["env"], {})
        for key, value in env_specific_config.items():
            default_config.env["eval"][key] = value

    default_config.lock()  # Make config read-only
  
    main(default_config)
