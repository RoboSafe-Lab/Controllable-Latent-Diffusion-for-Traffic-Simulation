
import  yaml,os,sys,argparse
import pytorch_lightning as pl
from my_registry import  get_my_registered_experiment_config
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env,set_global_trajdata_batch_raster_cfg
import tbsim.utils.train_utils as TrainUtils
from tbsim.datasets.factory import datamodule_factory
def load_yaml_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(cfg,auto_remove_exp_dir=False):
    pl.seed_everything(cfg.seed)
    set_global_batch_type("trajdata")
    set_global_trajdata_batch_env("nusc_trainval")
    set_global_trajdata_batch_raster_cfg(cfg.env.rasterizer)

    root_dir, log_dir, ckpt_dir, video_dir, version_key = TrainUtils.get_exp_dir(
        exp_name=cfg.name,
        output_dir=cfg.root_dir,
        save_checkpoints=cfg.train.save.enabled,
        auto_remove_exp_dir=auto_remove_exp_dir
    )
    cfg.dump(os.path.join(root_dir, version_key, "config.json"))
    train_callbacks = []

    # Training strategy: on single or multiple GPU
    assert cfg.train.parallel_strategy in ["dp","ddp_spawn",None,]
    # if only one GPU then override the locked config
    if not cfg.devices.num_gpus > 1:
        with cfg.train.unlocked():
            cfg.train.parallel_strategy = None
    if cfg.train.parallel_strategy in ["ddp_spawn"]:
        with cfg.train.training.unlocked():
            cfg.train.training.batch_size = int(
                cfg.train.training.batch_size / cfg.devices.num_gpus
            )
        with cfg.train.validation.unlocked():
            cfg.train.validation.batch_size = int(
                cfg.train.validation.batch_size / cfg.devices.num_gpus
            )

    datamodule = datamodule_factory(
        cls_name=cfg.train.datamodule_class, config=cfg
    )

    datamodule.setup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="../configs/config.yaml",
    )

    args = parser.parse_args()

    yaml_config = load_yaml_config(args.config)
    default_config = get_my_registered_experiment_config(yaml_config["config_name"])

    print("***** default_config *****", default_config)



    default_config.train.dataset_path = yaml_config["dataset_path"]
    default_config.root_dir = os.path.abspath(yaml_config["output_dir"])
    default_config.train.trajdata_source_train = yaml_config["trajdata_source_train"]
    default_config.train.trajdata_source_valid = yaml_config["trajdata_source_valid"]
    default_config.train.trajdata_data_dirs = yaml_config["trajdata_data_dirs"]
    if default_config.train.rollout.enabled:
        default_config.eval.env = default_config.env.name
        assert default_config.algo.eval_class is not None, \
            "Please set an eval_class for {}".format(default_config.algo.name)
        default_config.eval.eval_class = default_config.algo.eval_class
        default_config.eval.dataset_path = default_config.train.dataset_path
        for k in default_config.eval[default_config.eval.env]:  # copy env-specific config to the global-level
            default_config.eval[k] = default_config.eval[default_config.eval.env][k]
        default_config.eval.pop("nusc")
        default_config.eval.pop("l5kit")

    default_config.lock()  # Make config read-only
    main(default_config, auto_remove_exp_dir= yaml_config["remove_exp_dir"])