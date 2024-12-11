from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tbsim.models.diffuser import DiffuserModel
from tbsim.models.base_models import MLP
from tbsim.models.vaes import CVAE, FixedGaussianPrior
from tbsim.models.base_models import RNNTrajectoryEncoder,TrajectoryDecoder
import tbsim.utils.tensor_utils as TensorUtils
from typing import Dict, Any
import numpy as np
from tbsim.models.diffuser_helpers import unicyle_forward_dynamics
from tbsim.models.base_models import SplitMLP
class Encoder(nn.Module):
    def __init__(
            self,
            trajectory_shape: tuple,  # [T, D]
            output_shapes: OrderedDict,
            mlp_layer_dims: tuple = (128, 128),
            rnn_hidden_size: int = 100,
            normalization=False,
    ) -> None:
        super(Encoder, self).__init__()
        self.trajectory_shape = trajectory_shape

        # TODO: history encoder
        self.traj_encoder = RNNTrajectoryEncoder(
            trajectory_dim=trajectory_shape[-1],
            rnn_hidden_size=rnn_hidden_size
        )
        self.mlp = SplitMLP(
            input_dim=rnn_hidden_size ,
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims,
            output_activation=nn.ReLU,
            normalization=normalization,
        )

    def forward(self, inputs) -> Dict[str, torch.Tensor]:
        traj_feat = self.traj_encoder(inputs["trajectories"])#[B,100]
        return self.mlp(traj_feat)
class Decoder(TrajectoryDecoder):
    def __init__(self, feature_dim, state_dim, num_steps, step_time, dynamics_type, dynamics_kwargs, diffuser_norm_info):
        super(Decoder, self).__init__(
            feature_dim=feature_dim,
            state_dim=state_dim,
            num_steps=num_steps,
            step_time=step_time,
            dynamics_type=dynamics_type,
            dynamics_kwargs=dynamics_kwargs,
         )
        norm_add_coeffs = diffuser_norm_info[0]
        norm_div_coeffs = diffuser_norm_info[1]
        self.add_coeffs = np.array(norm_add_coeffs).astype('float32')
        self.div_coeffs = np.array(norm_div_coeffs).astype('float32')

    def _create_networks(self):
        hidden_dim = 256
        self.output_dim = self.dyn.udim #2    #self.num_steps * self.dyn.udim # 52*2
        self.z_to_h = nn.Linear(self.feature_dim, hidden_dim)
        self.z_to_c = nn.Linear(self.feature_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.output_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.h_to_out = nn.Linear(hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        # self.output_layer = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.4),
        #     nn.Linear(hidden_dim // 2, self.output_dim)  # 映射到动作维度
        # )
    def _forward_dynamics(self, current_states, actions):
        assert self.dyn is not None
        assert current_states.shape[-1] == self.dyn.xdim
        assert actions.shape[-1] == self.dyn.udim
        assert isinstance(self.step_time, float) and self.step_time > 0
        x_out_state = unicyle_forward_dynamics(
            dyn_model=self.dyn,
            initial_states=current_states,
            actions=actions,
            step_time=self.step_time
        )#x(B,52,4) state
        x_out_all = torch.cat([x_out_state, actions], dim=-1)

        #traj = torch.cat((pos, yaw), dim=-1)#(B,52,3--pos:2+yaw:1)
        return x_out_all#traj,x
    def _forward_networks(self, inputs, current_states=None, num_steps=None):
        # input:(B,64) z:64
        batch_size = inputs.size(0)
        h0 = torch.tanh(self.z_to_h(inputs)).unsqueeze(0)  # [1, B, hidden_dim]
        c0 = torch.tanh(self.z_to_c(inputs)).unsqueeze(0)  # [1, B, hidden_dim]
        zero_input = torch.zeros(batch_size, self.num_steps, self.output_dim, device=inputs.device)#[B,52,2]
        lstm_out, _ = self.lstm(zero_input, (h0, c0))  # [B, 52, 128]
        hidden_state = self.relu(lstm_out)
        output = self.h_to_out(hidden_state)  # [B, num_steps, output_dim]

        # num_steps = num_steps or self.num_steps #52
        # #action_dim = self.dyn.udim #2
        # decoder_inputs = inputs.unsqueeze(1).repeat(1, num_steps, 1)
        # lstm_out, _ = self.lstm(decoder_inputs)#(B,52,128)
        # actions = self.output_layer(lstm_out)#(B,52,2)
        preds = {
            "pred_action": output  # pred_action 表示动作序列
        }
        return preds
    def forward(self, inputs, current_states=None, num_steps=None, with_guidance=False, data_batch=None, num_samp=1):
        preds = self._forward_networks(inputs, current_states=current_states, num_steps=num_steps)
        if self.dyn is not None:
            if with_guidance:#未执行
                def decoder_wrapper(controls):
                    if len(current_states.shape) == 3:
                        current_states_join = TensorUtils.join_dimensions(current_states, 0, 2)
                    else:
                        current_states_join = current_states
                    trajectories, _ = self._forward_dynamics(current_states=current_states_join, actions=controls)
                    return trajectories
                if len(preds["pred_action"].shape) == 4:
                    preds_trajectories_join = TensorUtils.join_dimensions(preds["trajectories"], 0, 2)  # [B*N, T, 3]
                else:
                    preds_trajectories_join = preds["pred_action"]
                controls, _ = self.current_perturbation_guidance.perturb(
                    preds_trajectories_join, data_batch, self.guidance_optimization_params, num_samp=num_samp,
                    decoder=decoder_wrapper)
                if len(preds["pred_action"].shape) == 4:
                    controls = controls[:, None, ...]
                preds["controls"] = controls
            else:
                descale_action = self.descale_traj(preds["pred_action"],[4,5])
                #preds["controls"] = descale_action
            preds["trajectories"] = self._forward_dynamics(
                current_states=current_states,#(B,4)
                actions=descale_action,#(B,52,2)
            )#x(B,52,4) state算出来的; preds["trajectories"]:(B,52,3-- pos:2+yaw:1)
            rescale_x = self.scale_traj(preds["trajectories"] ,[0,1,2,3,4,5])

            preds["trajectories"] = rescale_x
        return preds
    def descale_traj(self, target_traj_orig, chosen_inds=[]):
        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D
        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device)
        target_traj = target_traj_orig * dx_div - dx_add
        return target_traj
    def scale_traj(self, target_traj_orig, chosen_inds=[]):
        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D
        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device)
        target_traj = (target_traj_orig + dx_add) / dx_div
        return target_traj

class HF_CVAE(nn.Module):
    def __init__(
        self,
        step_time:int,
        trajectory_shape: tuple,  # [T, D]
        latent_dim: int,
        dynamics_type: str,
        diffuser_norm_info: tuple,
        dynamics_kwargs=None,
        mlp_layer_dims: tuple = (128, 128),
        rnn_hidden_size: int = 256,
        kl_loss_weight: float = 1.0,
        recon_loss_weight: float = 1.0,
    ):
        super(HF_CVAE, self).__init__()
        self.encoder = Encoder(
            trajectory_shape=trajectory_shape,
            output_shapes=OrderedDict(mu=(latent_dim,), logvar=(latent_dim,)),
            mlp_layer_dims=mlp_layer_dims,
            rnn_hidden_size=rnn_hidden_size,
        )
        self.decoder = Decoder(
            feature_dim=latent_dim,
            state_dim=trajectory_shape[-1],
            num_steps=trajectory_shape[0],
            dynamics_type=dynamics_type,
            dynamics_kwargs=dynamics_kwargs,
            step_time=step_time,
            diffuser_norm_info=diffuser_norm_info,
        )
      
        self.latent_dim = latent_dim
        self.kl_loss_weight = kl_loss_weight
        self.recon_loss_weight = recon_loss_weight

    def forward(self, inputs: torch.Tensor,aux_info) -> Dict[str, Any]:
        #condition_features=None#aux_info['cond_feat']
        encoder_output = self.encoder(inputs={'trajectories': inputs})#{"mu":(B,128),"logvar":(B,128)}
        z=self.reparameterize(encoder_output['mu'],encoder_output['logvar'])
        #z = self.prior.sample_with_parameters(encoder_output, n=1).squeeze(dim=1)#(B,128)


        initial_states = aux_info['curr_states'] #(B,4)
        #decoder_input = torch.cat([z, condition_features], dim=-1)#(B,64+256)->(B,320)

        decoder_output = self.decoder(
            inputs=z,
            current_states=initial_states,
            num_steps=52
        )
        return {"decoder_output": decoder_output, "encoder_output": encoder_output, "z": z}

    def reparameterize(self,mu, logvar):
        # mu, logvar shape: [B, latent_dim]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)      
        return mu + eps * std







