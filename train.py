import os,yaml,argparse
import pytorch_lightning as pl
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from datetime import  datetime
from configs.custom_config import dict_to_config,ConfigBase
from tbsim.configs.base import ExperimentConfig
from utils.trainer_utils import prepare_trainer_and_data,prepare_for_guided_dm


def train_vae(cfg,debug=False):
    trainer, datamodule,model,ckpt_vae,_ = prepare_trainer_and_data(cfg,train_mode="vae",debug=cfg.train.debug)
    trainer.fit(model=model, datamodule=datamodule,ckpt_path=ckpt_vae)

def train_dm(cfg,debug=False):
    trainer, datamodule,model,ckpt_dm = prepare_trainer_and_data(cfg,train_mode="dm",debug=cfg.train.debug)
    trainer.fit(model=model, datamodule=datamodule,ckpt_path=ckpt_dm)

def train_guide_dm(cfg,debug=False):
    trainer, datamodule,model, ckpt = prepare_for_guided_dm(cfg,debug=cfg.train.debug)
    import torch
    from tbsim.utils.batch_utils import batch_utils
    from tbsim.models.diffuser_helpers import convert_state_to_state_and_action
    dataloader = datamodule.train_dataloader()


    total_sum = None
    total_sum_sq = None
    total_count = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch_utils().parse_batch(batch) 
            traj_state = torch.cat(
                (batch["target_positions"][:, :52, :], batch["target_yaws"][:, :52, :]), dim=2)
            traj_state_and_action = convert_state_to_state_and_action(traj_state, batch["curr_speed"], 0.1)
            B, T, D = traj_state_and_action.shape
            data = traj_state_and_action.reshape(-1, D)

            if total_sum is None:
                total_sum = data.sum(dim=0)
                total_sum_sq = (data ** 2).sum(dim=0)
            else:
                total_sum += data.sum(dim=0)
                total_sum_sq += (data ** 2).sum(dim=0)
            total_count += data.size(0)
            if total_count%100==0:
                print(total_count)
    mean = total_sum / total_count
    var = total_sum_sq / total_count - mean ** 2
    std = torch.sqrt(var)
    print(f"mean:{mean}, std:{std}")
    #mean:tensor([ 3.9122e+00, -1.4352e-03,  1.5173e+00,  7.2863e-05,  8.3396e-03, 6.0973e-05]), std:tensor([9.5063, 1.5163, 3.1296, 0.1343, 2.0385, 0.0574])
    # trainer.fit(model=model,datamodule=datamodule,ckpt_path=ckpt)

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
    elif cfg.train.mode == 'guide_dm':
        train_guide_dm(cfg)
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
