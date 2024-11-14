import os
import argparse
from dataclasses import asdict
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from configs.experiment_config import ExperimentConfig, EvalConfig
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
import tbsim.utils.train_utils as TrainUtils
from tbsim.datasets.factory import datamodule_factory


def replace_infinity(cfg: DictConfig):

    def recursive_replace(d):
        for k, v in d.items():
            if isinstance(v, str) and v.lower() == "infinity":
                d[k] = float('inf')
            elif isinstance(v, DictConfig):
                recursive_replace(v)
            elif isinstance(v, list):
                for idx, item in enumerate(v):
                    if isinstance(item, str) and item.lower() == "infinity":
                        v[idx] = float('inf')
                    elif isinstance(item, DictConfig):
                        recursive_replace(item)

    recursive_replace(cfg)


def main(cfg: DictConfig, auto_remove_exp_dir=False):

    seed = cfg.train.training_num_steps if hasattr(cfg.train, 'training_num_steps') else 42
    pl.seed_everything(seed)


    set_global_batch_type("trajdata")
    set_global_trajdata_batch_env(cfg.env.data_generation_trajdata_centric)
    set_global_trajdata_batch_raster_cfg(cfg.env.rasterizer)


    root_dir, log_dir, ckpt_dir, video_dir, version_key = TrainUtils.get_exp_dir(
        exp_name=cfg.registered_name,
        output_dir=cfg.output_dir,
        save_checkpoints=cfg.train.save_best_k > 0,
        auto_remove_exp_dir=auto_remove_exp_dir
    )


    config_save_path = os.path.join(root_dir, version_key, "config.yaml")
    OmegaConf.save(cfg, config_save_path)
    print(f"Configuration saved to {config_save_path}")

    train_callbacks = []


    assert cfg.train.parallel_strategy in ["dp", "ddp_spawn", None], "Invalid parallel strategy"


    num_gpus = OmegaConf.to_container(cfg, resolve=True).get("devices", {}).get("num_gpus", 1)
    if num_gpus <= 1:
        cfg.train.parallel_strategy = None

    if cfg.train.parallel_strategy == "ddp_spawn":
        cfg.train.training_batch_size = int(cfg.train.training_batch_size / num_gpus)
        cfg.train.validation_batch_size = int(cfg.train.validation_batch_size / num_gpus)


    datamodule = datamodule_factory(
        cls_name=cfg.train.datamodule_class, config=cfg
    )

    datamodule.setup()


    # model = YourModel(cfg.algo)
    # trainer = pl.Trainer(...)
    # trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="../config.yaml",
        help="Path to the main configuration YAML file"
    )
    args = parser.parse_args()


    cfg = OmegaConf.structured(ExperimentConfig)
    yaml_config = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, yaml_config)


    replace_infinity(cfg)


    if cfg.train.rollout_enabled:
        if cfg.eval is None:
            cfg.eval = EvalConfig()
        cfg.eval.env = cfg.env.data_generation_trajdata_centric
        cfg.eval.eval_class = cfg.algo.eval_class
        cfg.eval.dataset_path = cfg.dataset_path



    print("***** Loaded Configuration *****")
    print(OmegaConf.to_yaml(cfg))


    main(cfg, auto_remove_exp_dir=cfg.get("remove_exp_dir", False))