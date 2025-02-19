import yaml,argparse
import pytorch_lightning as pl
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from datetime import  datetime
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
    # import torch
    # from tbsim.utils.batch_utils import batch_utils
    # from tbsim.models.diffuser_helpers import convert_state_to_state_and_action
    # from models.context_utils import get_state_and_action_from_data_batch
    # dataloader = datamodule.train_dataloader()


    # total_sum = torch.zeros(6)          # 用于存储所有数据中每个特征的和，初始值为 [0,0,0,0,0,0]
    # total_sq_sum = torch.zeros(6)       # 用于存储所有数据中每个特征的平方和，初始值为 [0,0,0,0,0,0]
    # total_count = 0

    # i = 0
    # with torch.no_grad():
    #     for batch in dataloader:
    #         batch = batch_utils().parse_batch(batch) 
    #         state_and_action = get_state_and_action_from_data_batch(batch)
    #         B, T, _ = state_and_action.shape 

    #         total_sum += state_and_action.sum(dim=(0, 1)) 
    #         total_sq_sum += (state_and_action ** 2).sum(dim=(0, 1))
    #         total_count += B * T
    # mean = total_sum / total_count
    # variance = total_sq_sum / total_count - mean ** 2  
    # std = torch.sqrt(variance) 

    # print("Dataset feature mean:", mean)
    # print("Dataset feature std:", std)  
    # def create_wandb_dir(base_dir="logs"):
    #     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     run_dir = os.path.join(base_dir, timestamp)
    #     os.makedirs(run_dir, exist_ok=True)
    #     return run_dir
