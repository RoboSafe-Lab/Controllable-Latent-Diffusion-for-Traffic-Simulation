from dataclasses import dataclass, field

@dataclass
class NuscTrajdataTrainConfig:
    trajdata_cache_location: str = "~/.unified_data_cache"
    trajdata_source_train: list = field(default_factory=lambda: ["nusc_mini-mini_train"])
    trajdata_source_valid: list = field(default_factory=lambda: ["nusc_mini-mini_val"])
    trajdata_data_dirs: dict = field(default_factory=lambda: {
        "nusc_trainval": "../behavior-generation-dataset/nuscenes",
        "nusc_test": "../behavior-generation-dataset/nuscenes",
        "nusc_mini": "../behavior-generation-dataset/nuscenes/mini",
    })
    trajdata_rebuild_cache: bool = False

    # Rollout configurations
    rollout_enabled: bool = True
    rollout_save_video: bool = True
    rollout_every_n_steps: int = 10000
    rollout_warm_start_n_steps: int = 0

    # Training configurations
    training_batch_size: int = 4
    training_num_steps: int = 1000
    training_num_data_workers: int = 8

    # Saving configurations
    save_every_n_steps: int = 10000
    save_best_k: int = 10

    # Validation configurations
    validation_enabled: bool = True
    validation_batch_size: int = 4
    validation_num_data_workers: int = 0
    validation_every_n_steps: int = 500
    validation_num_steps_per_epoch: int = 50

    # Other configurations
    on_ngc: bool = False

    # Logging configurations
    logging_terminal_output_to_txt: bool = True
    logging_log_tb: bool = False
    logging_log_wandb: bool = False
    logging_wandb_project_name: str = "tbsim"
    logging_log_every_n_steps: int = 10
    logging_flush_every_n_steps: int = 100