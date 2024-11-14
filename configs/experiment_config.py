from dataclasses import dataclass, field
from typing import Optional
from configs.train_config import NuscTrajdataTrainConfig
from configs.env_config import NuscTrajdataEnvConfig
from configs.algo_config import DiffuserConfig

@dataclass
class EvalConfig:
    env: str = ""
    eval_class: str = ""
    dataset_path: str = ""

@dataclass
class ExperimentConfig:

    train: NuscTrajdataTrainConfig
    env: NuscTrajdataEnvConfig
    algo: DiffuserConfig


    config_name: str = "custom_trajdata_diff_xyz"
    registered_name: str = "custom_trajdata_diff_xyz"
    dataset_path: str = ""
    remove_exp_dir: bool = True
    output_dir: str = "../logs"
    trajdata_source_train: list = field(default_factory=lambda: ["nusc_mini-mini_train"])
    trajdata_source_valid: list = field(default_factory=lambda: ["nusc_mini-mini_val"])
    trajdata_data_dirs: dict = field(default_factory=lambda: {
        "nusc_trainval": "../behavior-generation-dataset/nuscenes",
        "nusc_test": "../behavior-generation-dataset/nuscenes",
        "nusc_mini": "../behavior-generation-dataset/nuscenes/mini",
    })
    eval: Optional[EvalConfig] = field(default=None)