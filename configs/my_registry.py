from tbsim.configs.base import ExperimentConfig
from my_config import MyCustomTrainConfig,MyCustomEnvConfig
from tbsim.configs.algo_config import DiffuserConfig

EXP_CONFIG_REGISTRY = dict()
EXP_CONFIG_REGISTRY["trajdata_nusc_diff_xyz"] = ExperimentConfig(
    train_config=MyCustomTrainConfig(),
    env_config=MyCustomEnvConfig(),
    algo_config=DiffuserConfig(),
    registered_name="trajdata_nusc_diff_xyz"
)

def get_my_registered_experiment_config(registered_name):

    if registered_name not in EXP_CONFIG_REGISTRY.keys():
        raise KeyError(
            "'{}' is not a registered experiment config please choose from {}".format(
                registered_name, list(EXP_CONFIG_REGISTRY.keys())
            )
        )
    return EXP_CONFIG_REGISTRY[registered_name].clone()