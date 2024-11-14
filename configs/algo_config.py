from dataclasses import dataclass, field
import math
from typing import Optional

@dataclass
class DiffuserConfig:
    eval_class: str = "Diffuser"
    name: str = "diffuser"
    coordinate: str = 'agent_centric'  # ['agent_centric', 'scene_centric']
    scene_agent_max_neighbor_dist: float = 30
    map_encoder_model_arch: str = "resnet18"
    model_architecture: str = "TemporalMapUnet"
    disable_control_on_stationary: str = 'current_speed'
    moving_speed_th: float = 0.5
    rasterized_history: bool = True
    rasterized_map: bool = True
    use_map_feat_global: bool = True
    use_map_feat_grid: bool = False
    base_dim: int = 32
    horizon: int = 52
    n_diffusion_steps: int = 100
    action_weight: int = 1
    diffusor_loss_weights: Optional[float] = None  # 使用 Optional 允许为 None
    loss_discount: int = 1
    predict_epsilon: bool = False
    dim_mults: tuple = (2, 4, 8)
    clip_denoised: bool = False
    loss_type: str = 'l2'
    use_ema: bool = True
    ema_step: int = 1
    ema_decay: float = 0.995
    ema_start_step: int = 4000
    diffuser_building_block: str = 'concat'
    action_loss_only: bool = False
    diffuser_input_mode: str = 'state_and_action'  # ['state', 'action', 'state_and_action', 'state_and_action_no_dyn']
    use_reconstructed_state: bool = False
    conditioning_drop_map_p: float = 0.0
    conditioning_drop_neighbor_p: float = 0.0
    conditioning_drop_fill: float = 0.5
    cond_feat_dim: int = 256
    curr_state_feat_dim: int = 64
    map_feature_dim: int = 256
    map_grid_feature_dim: int = 32
    history_feature_dim: int = 128
    history_num_frames: int = 30
    history_num_frames_ego: int = 30
    history_num_frames_agents: int = 30
    future_num_frames: int = 52
    step_time: float = 0.1
    render_ego_history: bool = False
    decoder_layer_dims: tuple = ()
    decoder_state_as_input: bool = True
    dynamics_type: str = "Unicycle"
    dynamics_max_steer: float = 0.5
    dynamics_max_yawvel: float = math.pi * 2.0
    dynamics_acce_bound: tuple = (-10, 8)
    dynamics_ddh_bound: tuple = (-math.pi * 2.0, math.pi * 2.0)
    dynamics_max_speed: float = 40.0
    loss_weights_diffusion_loss: float = 1.0
    optim_params_policy_learning_rate_initial: float = 2e-4
    optim_params_policy_learning_rate_decay_factor: float = 0.1
    optim_params_policy_learning_rate_epoch_schedule: list = field(default_factory=list)
    diffuser_num_eval_samples: int = 10
    nusc_norm_info: dict = field(default_factory=lambda: {
        'diffuser': [
            [2.135494, 0.003704, 0.970226, 0.000573, -0.002965, 0.000309],
            [5.5444, 0.524067, 2.206522, 0.049049, 0.729327, 0.023765]
        ],
        'agent_hist': [
            [-1.198923, 0.000128, 0.953161, 4.698113, 2.051664],
            [3.180241, 0.159182, 2.129779, 2.116855, 0.388149]
        ],
        'neighbor_hist': [
            [-0.237441, 1.118636, 0.489575, 0.868664, 0.222984],
            [7.587311, 7.444489, 1.680952, 2.578202, 0.832563]
        ]
    })
    nuplan_norm_info: dict = field(default_factory=lambda: {
        'diffuser': [
            [4.796822, 0.005868, 2.192235, 0.001452, 0.031407, 0.000549],
            [9.0485, 0.502242, 3.416214, 0.081688, 5.372146, 0.074966]
        ],
        'agent_hist': [
            [-2.740177, 0.002556, 2.155241, 3.783593, 1.812939],
            [5.212748, 0.139311, 3.422815, 1.406989, 0.298052]
        ],
        'neighbor_hist': [
            [-0.405011, 1.321956, 0.663815, 0.581748, 0.162458],
            [8.044137, 6.039638, 2.541506, 1.919543, 0.685904]
        ]
    })