from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tbsim.models.diffuser import DiffuserModel
from tbsim.models.base_models import MLP
from tbsim.models.vaes import CVAE, FixedGaussianPrior
from tbsim.models.base_models import PosteriorEncoder
from tbsim.models.base_models import TrajectoryDecoder
import tbsim.utils.tensor_utils as TensorUtils
from typing import Dict, Any
class HF_Decoder(TrajectoryDecoder):
    def __init__(self, feature_dim,state_dim,num_steps,step_time,dynamics_type,dynamics_kwargs):
        super(HF_Decoder, self).__init__(feature_dim=feature_dim,
                                         state_dim=state_dim,
                                         num_steps=num_steps,
                                         step_time=step_time,
                                         dynamics_type=dynamics_type,
                                         dynamics_kwargs=dynamics_kwargs,
                                         )

    def _create_networks(self):

        input_dim = self.feature_dim  # z  + condition_features
        output_dim = self.num_steps * self.dyn.udim
        hidden_dim = 128
        self.decoder_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def _forward_networks(self, inputs, current_states=None, num_steps=None):
        # input:(B,320)
        batch_size = inputs.size(0)
        num_steps = num_steps or self.num_steps #52
        action_dim = self.dyn.udim


        actions = self.decoder_mlp(inputs)  # [batch_size, num_steps * action_dim](B,52*2)


        actions = actions.view(batch_size, num_steps, action_dim)#(B,52,2)

        preds = {
            "trajectories": actions  # trajectories 表示动作序列
        }
        return preds
    def forward(self, inputs, current_states=None, num_steps=None, with_guidance=False, data_batch=None, num_samp=1):

        preds = self._forward_networks(inputs, current_states=current_states, num_steps=num_steps)
        if self.dyn is not None:
            if with_guidance:
                def decoder_wrapper(controls):
                    if len(current_states.shape) == 3:
                        current_states_join = TensorUtils.join_dimensions(current_states, 0, 2)
                    else:
                        current_states_join = current_states
                    trajectories, _ = self._forward_dynamics(current_states=current_states_join, actions=controls)
                    return trajectories


                if len(preds["trajectories"].shape) == 4:
                    preds_trajectories_join = TensorUtils.join_dimensions(preds["trajectories"], 0, 2)  # [B*N, T, 3]
                else:
                    preds_trajectories_join = preds["trajectories"]


                controls, _ = self.current_perturbation_guidance.perturb(
                    preds_trajectories_join, data_batch, self.guidance_optimization_params, num_samp=num_samp,
                    decoder=decoder_wrapper)


                if len(preds["trajectories"].shape) == 4:
                    controls = controls[:, None, ...]
                preds["controls"] = controls
            else:
                preds["controls"] = preds["trajectories"]


            preds["trajectories"], x = self._forward_dynamics(
                current_states=current_states,#(B,4)
                actions=preds["controls"]#(B,52,2)
            )
            preds["terminal_state"] = x[..., -1, :]
            preds["states"] = x
        return preds


class HF_CVAE(nn.Module):
    def __init__(
        self,
        step_time:int,
        trajectory_shape: tuple,  # [T, D]
        condition_dim: int,
        latent_dim: int,
        dynamics_type: str,
        dynamics_kwargs=None,
        mlp_layer_dims: tuple = (128, 128),
        rnn_hidden_size: int = 100,
        kl_loss_weight: float = 1.0,
        recon_loss_weight: float = 1.0,

    ):
        super(HF_CVAE, self).__init__()
        self.q_net = PosteriorEncoder(
            condition_dim=condition_dim,
            trajectory_shape=trajectory_shape,
            output_shapes=OrderedDict(mu=(latent_dim,), logvar=(latent_dim,)),
            mlp_layer_dims=mlp_layer_dims,
            rnn_hidden_size=rnn_hidden_size,
        )
        # Decoder

        self.decoder = HF_Decoder(
            feature_dim=latent_dim + condition_dim,
            state_dim=trajectory_shape[-1],
            num_steps=trajectory_shape[0],
            dynamics_type=dynamics_type,
            dynamics_kwargs=dynamics_kwargs,
            step_time=step_time,

        )


        self.prior = FixedGaussianPrior(latent_dim=latent_dim)



        self.latent_dim = latent_dim
        self.kl_loss_weight = kl_loss_weight
        self.recon_loss_weight = recon_loss_weight

    def forward(self, inputs: torch.Tensor,condition_features,decoder_kwargs=None) -> Dict[str, Any]:
        encoder_output = self.q_net(inputs={'trajectories': inputs}, condition_features=condition_features)#{"mu":(B,64),"logvar":(B,64)}
        z = self.prior.sample_with_parameters(encoder_output, n=1).squeeze(dim=1)#(B,64)
        initial_states = inputs[:, 0, :]# (B,52,6)->(B,6)初始状态，采用52个时间步的第一个
        decoder_input = torch.cat([z, condition_features], dim=-1)#(B,64+256)->(B,320)
        current_states = initial_states[:, :4]#(B,4)
        decoder_output = self.decoder(
            inputs=decoder_input,
            current_states=current_states,
            num_steps=52
        )
        return {"decoder_output": decoder_output, "encoder_output": encoder_output, "z": z, "condition_features": condition_features}



