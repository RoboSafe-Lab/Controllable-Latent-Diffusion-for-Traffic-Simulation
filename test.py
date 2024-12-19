from tbsim.utils.scene_edit_utils import UnifiedRenderer
import numpy as np

import argparse
import  sys
import pytorch_lightning as pl

from tbsim.utils.log_utils import PrintLogger
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
import tbsim.utils.train_utils as TrainUtils
from tbsim.datasets.factory import datamodule_factory
from tbsim.utils.env_utils import RolloutCallback

from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata.visualization.vis import plot_agent_batch
import wandb,json
from pytorch_lightning.loggers import  WandbLogger
from  models.algos import  UnifiedTrainer
from datetime import  datetime
from configs.custom_config import dict_to_config,ConfigBase,serialize_object
from src.tbsim.configs.base import ExperimentConfig
import yaml
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.augmentation import NoiseHistories
from trajdata.visualization.vis import plot_agent_batch,plot_agent_batch_all
import matplotlib
matplotlib.use('TkAgg')
parser = argparse.ArgumentParser(description="Training Script")
parser.add_argument("--config", type=str, default="/home/visier/hazardforge/HazardForge/config.yaml", help="Path to YAML config")

# 直接使用默认值解析参数
args = parser.parse_args([])  # 在 Notebook 中使用 [] 表示不传递命令行参数

# 加载配置文件
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
default_config.lock()
cfg = default_config
pl.seed_everything(cfg.seed)
set_global_batch_type("trajdata")
set_global_trajdata_batch_env(cfg.train.trajdata_source_train[0])
set_global_trajdata_batch_raster_cfg(cfg.env.rasterizer)
print("\n============= New Training Run with Config =============")

datamodule = datamodule_factory(
        cls_name=cfg.train.datamodule_class, config=cfg
    )
datamodule.setup()


dataset = datamodule.train_dataset
print(f"# Data Samples: {len(dataset):,}")
dataloader = DataLoader(
        dataset,
        batch_size=4,

        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=4,
    )
import matplotlib
matplotlib.use('TkAgg') 
batch: AgentBatch
 # Enable interactive mode
for batch in tqdm(dataloader):
    plot_agent_batch(batch, batch_idx=0)
    # plot_agent_batch_all(batch)
     # Displays the plot
   