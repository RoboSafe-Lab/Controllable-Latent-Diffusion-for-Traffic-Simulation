from tbsim.datasets.trajdata_datamodules import PassUnifiedDataModule
from trajdata import UnifiedDataset
from tbsim.utils.trajdata_utils import TRAJDATA_AGENT_TYPE_MAP, get_closest_lane_point_wrapper, get_full_fut_traj, get_full_fut_valid
from collections import defaultdict
import gc
import os
from torch.utils.data import DataLoader
class Hf_DataModule(PassUnifiedDataModule):
    def setup(self, stage = None):
        data_cfg = self._data_config
        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance
        agent_only_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_only_types]
        agent_predict_types = None
        print("data_cfg.trajdata_predict_types", data_cfg.trajdata_predict_types)
        if data_cfg.trajdata_predict_types is not None:
            if data_cfg.other_agents_num is None:
                max_agent_num = None
            else:
                max_agent_num = 1+data_cfg.other_agents_num

            agent_predict_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_predict_types]
        kwargs = dict(
            cache_location=data_cfg.trajdata_cache_location,
            desired_data=data_cfg.trajdata_source_train,
            desired_dt=data_cfg.step_time,
            future_sec=(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            data_dirs=data_cfg.trajdata_data_dirs,
            only_types=agent_only_types, # vehicle
            only_predict=agent_predict_types, # vehicle
            agent_interaction_distances=defaultdict(lambda: neighbor_distance), #20
            incl_raster_map=data_cfg.trajdata_incl_map,
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center,
                "no_map_fill_value": data_cfg.no_map_fill_value,
            },
            incl_vector_map=True,
            centric=data_cfg.trajdata_centric,
            scene_description_contains=data_cfg.trajdata_scene_desc_contains,
            standardize_data=data_cfg.trajdata_standardize_data,
            verbose=True,
            # max_agent_num = max_agent_num,
            num_workers=os.cpu_count(),
            rebuild_cache=data_cfg.trajdata_rebuild_cache,
            rebuild_maps=data_cfg.trajdata_rebuild_cache,
            # A dictionary that contains functions that generate our custom data.
            # Can be any function and has access to the batch element.
            extras={
                "closest_lane_point": get_closest_lane_point_wrapper(self._train_config.training_vec_map_params),
                "full_fut_traj": get_full_fut_traj,
                "full_fut_valid": get_full_fut_valid,
            },
            ego_only = True,
            max_neighbor_num = 5,
            # incl_robot_future = True,
        )
        print(kwargs)
        self.train_dataset = UnifiedDataset(**kwargs)

        kwargs["desired_data"] = data_cfg.trajdata_source_valid
        self.valid_dataset = UnifiedDataset(**kwargs)

        # set modality shape based on input
        self.num_sem_layers = 0 if not data_cfg.trajdata_incl_map else data_cfg.num_sem_layers

        gc.collect()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self._train_config.training.batch_size,
            num_workers=self._train_config.training.num_data_workers,
            drop_last=True,
            collate_fn=self.train_dataset.get_collate_fn(return_dict=True),
            persistent_workers=True,

        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            shuffle=False, # since pytorch lightning only evals a subset of val on each epoch, shuffle
            batch_size=self._train_config.validation.batch_size,
            num_workers=self._train_config.validation.num_data_workers,
            drop_last=True,
            collate_fn=self.valid_dataset.get_collate_fn(return_dict=True),
            persistent_workers=True,

        )