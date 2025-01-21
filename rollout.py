import argparse
import numpy as np
import json
import random
import yaml
import importlib
import os
import torch
from tbsim.utils.tensor_utils import map_ndarray
import tbsim.utils.tensor_utils as TensorUtils
from models.rollout.scene_edit_config import Hf_SceneEditingConfig
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from tbsim.datasets.trajdata_datamodules import PassUnifiedDataModule
from configs.custom_config import dict_to_config,ConfigBase
from tbsim.configs.base import ExperimentConfig
from tbsim.utils.config_utils import translate_pass_trajdata_cfg
from trainers.dm_trainer import DMLightningModule
from tbsim.evaluation.env_builders import EnvUnifiedBuilder
from tbsim.utils.scene_edit_utils import get_trajdata_renderer
def run_scene_editor(eval_cfg,policy_cfg, save_cfg, data_to_disk, render_to_video, render_to_img, render_cfg):
    set_global_batch_type("trajdata")
    set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0])
    np.random.seed(eval_cfg.seed)
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)
    torch.cuda.manual_seed(eval_cfg.seed)
    print('saving results to {}'.format(eval_cfg.results_dir))
    os.makedirs(eval_cfg.results_dir, exist_ok=True)

    if render_to_video or render_to_img:
        os.makedirs(os.path.join(eval_cfg.results_dir, "viz/"), exist_ok=True)
    if save_cfg:
        json.dump(eval_cfg, open(os.path.join(eval_cfg.results_dir, "config.json"), "w+"),indent=4)
    if data_to_disk and os.path.exists(eval_cfg.experience_hdf5_path):
        os.remove(eval_cfg.experience_hdf5_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    trajdata_config = translate_pass_trajdata_cfg(policy_cfg)

    datamodule = PassUnifiedDataModule(trajdata_config, policy_cfg.train)
    datamodule.setup()
    policy = DMLightningModule.load_from_checkpoint(
        policy_cfg.train.checkpoint_dm,
        # '/home/visier/hazardforge/HazardForge/checkpoint/dm/loss.ckpt',
        algo_config=policy_cfg.algo,
        train_config=policy_cfg.train,
        modality_shapes=datamodule.modality_shapes,
        vae_model_path=policy_cfg.train.checkpoint_vae
    ).to(device).eval()
    set_global_trajdata_batch_raster_cfg(policy_cfg.env.rasterizer)
    env_builder = EnvUnifiedBuilder(eval_config=eval_cfg, exp_config=policy_cfg, device=device)
    env = env_builder.get_env()
    obs_to_torch= True
    render_rasterizer = get_trajdata_renderer(eval_cfg.trajdata_source_test,
                                                  eval_cfg.trajdata_data_dirs,
                                                  future_sec=eval_cfg.future_sec,
                                                  history_sec=eval_cfg.history_sec,
                                                  raster_size=render_cfg['size'],
                                                  px_per_m=render_cfg['px_per_m'],
                                                  rebuild_maps=False,
                                                  cache_location='~/my_custom_cache_location')
    


    result_stats = None
    scene_i = 0
    eval_scenes = eval_cfg.eval_scenes
    while scene_i < eval_cfg.num_scenes_to_evaluate:
        scene_indices = eval_scenes[scene_i: scene_i + eval_cfg.num_scenes_per_batch]
        scene_i += eval_cfg.num_scenes_per_batch
        print('scene_indices', scene_indices)

        scenes_valid = env.reset(scene_indices=scene_indices, start_frame_index=None)
        scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]
        if len(scene_indices) == 0:
            print('no valid scenes in this batch, skipping...')
            torch.cuda.empty_cache()
            continue
        
        start_frame_index = [[policy_cfg.algo.history_num_frames+1]] * len(scene_indices)
        print('Starting frames in current scenes:', start_frame_index)
        for ei in range(eval_cfg.num_sim_per_scene):
            guidance_config = None
            constraint_config = None
            cur_start_frames = [scene_start[ei] for scene_start in start_frame_index]
            scenes_valid = env.reset(scene_indices=scene_indices, start_frame_index=cur_start_frames)
            sim_scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]
            sim_start_frames = [sframe for sframe, sval in zip(cur_start_frames, scenes_valid) if sval]
            if len(sim_scene_indices) == 0:
                torch.cuda.empty_cache()
                continue

        done  = env.is_done()
        while not done:
            obs = env.get_observation()
            action = policy.get_action(obs)   # policy里包含dm+rl guidance
            env.step(action)
            done = env.is_done()











        print("1111111111111111111111111111111111111111111111")

  





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="A json file containing evaluation configs"
    )

    parser.add_argument(
        "--env",
        type=str,
        default= "trajdata",
        help="Which env to run editing in",
      
    )

    parser.add_argument(
        "--ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location of each model",
        default=None
    )

    parser.add_argument(
        "--metric_ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location for the learned metric",
        default=None
    )

    parser.add_argument(
        "--eval_class",
        type=str,
        default='DmComposer',
        help="Optionally specify the evaluation class through argparse"
    )

    parser.add_argument(
        "--policy_ckpt_dir",
        type=str,
        default='/home/visier/hazardforge/HazardForge/checkpoint/dm',
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--policy_ckpt_key",
        type=str,
        default='loss.ckpt',
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )
    # ------ for BITS ------
    parser.add_argument(
        "--planner_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--planner_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )

    parser.add_argument(
        "--predictor_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--predictor_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )
    # ----------------------

    parser.add_argument(
        "--results_root_dir",
        type=str,
        default='/home/visier/hazardforge/HazardForge/logs/rollout_results',
        help="Directory to save results and videos"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default='/home/visier/nuscenes',
        help="Root directory of the dataset"
    )

    parser.add_argument(
        "--num_scenes_per_batch",
        type=int,
        default=1,
        help="Number of scenes to run concurrently (to accelerate eval)"
    )

    parser.add_argument(
    "--no_render",
    dest="render",
    action="store_false",  
    help="Disable rendering."
)
    parser.set_defaults(render=True)

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--registered_name",
        type=str,
        default='trajdata_nusc_diff',
    )

    parser.add_argument(
        "--render_img",
        action="store_true",
        default=False,
        help="whether to only render the first frame of rollout"
    )

    parser.add_argument(
        "--render_size",
        type=int,
        default=224,
        help="width and height of the rendered image size in pixels"
    )

    parser.add_argument(
        "--render_px_per_m",
        type=float,
        default=2.0,
        help="resolution of rendering"
    )

    parser.add_argument(
        "--save_every_n_frames",
        type=int,
        default=5,
        help="saving videos while skipping every n frames"
    )

    parser.add_argument(
        "--draw_mode",
        type=str,
        default='action',
        help="['action', 'entire_traj', 'map']"
    )
    
    #
    # Editing options
    #
    parser.add_argument(
        "--editing_source",
        type=str,
        # choices=["config", "heuristic", "ui", "none"],
        default= 'heuristic',
        nargs="+",
        help="Which edits to use. config is directly from the configuration file. heuristic will \
              set edits automatically based on heuristics. UI will use interactive interface. \
              config and heuristic may be used together. If none, does not use edits."
    )
    parser.add_argument(
        "--policy_config",
        type=str,
        default='/home/visier/hazardforge/HazardForge/checkpoint/dm/config.json'
    )

    args = parser.parse_args()

    cfg = Hf_SceneEditingConfig(registered_name=args.registered_name)

    with open(args.policy_config,"r")as f:
        policy_dict = yaml.safe_load(f)
    train_config = dict_to_config(ConfigBase, policy_dict.get("train", {}))
    env_config = dict_to_config(ConfigBase, policy_dict.get("env", {}))
    algo_config = dict_to_config(ConfigBase, policy_dict.get("algo", {}))
    policy_config = ExperimentConfig(
        train_config=train_config,
        env_config=env_config,
        algo_config=algo_config,
        registered_name=policy_dict.get("registered_name", "default_experiment"),
    )
    policy_config.lock()

    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    if args.eval_class is not None:
        cfg.eval_class = args.eval_class

    if args.policy_ckpt_dir is not None:
        assert args.policy_ckpt_key is not None, "Please specify a key to look for the checkpoint, e.g., 'iter50000'"
        cfg.ckpt.policy.ckpt_dir = args.policy_ckpt_dir
        cfg.ckpt.policy.ckpt_key = args.policy_ckpt_key

    if args.planner_ckpt_dir is not None:
        cfg.ckpt.planner.ckpt_dir = args.planner_ckpt_dir
        cfg.ckpt.planner.ckpt_key = args.planner_ckpt_key

    if args.predictor_ckpt_dir is not None:
        cfg.ckpt.predictor.ckpt_dir = args.predictor_ckpt_dir
        cfg.ckpt.predictor.ckpt_key = args.predictor_ckpt_key

    if args.num_scenes_per_batch is not None:
        cfg.num_scenes_per_batch = args.num_scenes_per_batch

    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path

    if cfg.name is None:
        cfg.name = cfg.eval_class

    if args.prefix is not None:
        cfg.name = args.prefix + cfg.name

    if args.seed is not None:
        cfg.seed = args.seed
    if args.results_root_dir is not None:
        cfg.results_dir = os.path.join(args.results_root_dir, cfg.name)
    else:
        cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)
    
    if args.policy_config is not None:
        cfg.policy_config = args.policy_config
    # add eval_class into the results_dir
    # cfg.results_dir = os.path.join(cfg.results_dir, cfg.eval_class)

    if args.env is not None:
        cfg.env = args.env
    else:
        assert cfg.env is not None

    if args.editing_source is not None:
        cfg.edits.editing_source = args.editing_source
    if not isinstance(cfg.edits.editing_source, list):
        cfg.edits.editing_source = [cfg.edits.editing_source]
    if "ui" in cfg.edits.editing_source:
        # can only handle one scene with UI
        cfg.num_scenes_per_batch = 1

    cfg.experience_hdf5_path = os.path.join(cfg.results_dir, "data.hdf5")

    for k in cfg[cfg.env]:  # copy env-specific config to the global-level
        cfg[k] = cfg[cfg.env][k]

    cfg.pop("nusc")
    cfg.pop("trajdata")

    if args.ckpt_yaml is not None:
        with open(args.ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)
    if args.metric_ckpt_yaml is not None:
        with open(args.metric_ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)
    
    render_cfg = {
        'size' : args.render_size,
        'px_per_m' : args.render_px_per_m,
        'save_every_n_frames': args.save_every_n_frames,
        'draw_mode': args.draw_mode,
    }

    cfg.lock()
    run_scene_editor(
        cfg,
        policy_config,
        save_cfg=True,
        data_to_disk=True,
        render_to_video=args.render,
        render_to_img=args.render_img,
        render_cfg=render_cfg,
    )
