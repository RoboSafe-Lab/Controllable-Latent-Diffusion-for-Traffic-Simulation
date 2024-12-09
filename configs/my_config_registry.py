from src.tbsim.configs.base import ExperimentConfig

from configs.custom_config import HfCustomTrainConfig,HfCustomEnvConfig,HfCustomAlgoConfig


MY_EXP_CONFIG_REGISTRY = dict()

MY_EXP_CONFIG_REGISTRY["nusc_hf"] = ExperimentConfig(
    train_config=HfCustomTrainConfig(),
    env_config=HfCustomEnvConfig(),
    algo_config=HfCustomAlgoConfig(),
    registered_name="nusc_hf"
)


from src.tbsim.configs.registry import EXP_CONFIG_REGISTRY
MY_EXP_CONFIG_REGISTRY.update(EXP_CONFIG_REGISTRY)

def get_registered_experiment_config(registered_name):
    if registered_name not in MY_EXP_CONFIG_REGISTRY.keys():
        raise KeyError(
            "'{}' is not a registered experiment config please choose from {}".format(
                registered_name, list(EXP_CONFIG_REGISTRY.keys())
            )
        )
    return MY_EXP_CONFIG_REGISTRY[registered_name].clone()