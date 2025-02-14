
import torch.nn as nn
import numpy as np
from models.vae.lstm_vae import LSTMVAE
import torch
from tbsim.utils.batch_utils import batch_utils
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.dynamics as dynamics
from torch.nn import functional as F
from tbsim.models.diffuser_helpers import MapEncoder,convert_state_to_state_and_action,unicyle_forward_dynamics
from models.context_utils import ContextEncoder,get_state_and_action_from_data_batch
class VaeModel(nn.Module):
    def __init__(self, algo_config,train_config, modality_shapes):

        super(VaeModel, self).__init__()
        self.algo_config = algo_config
        self.train_config = train_config
        self.data_centric = None
        self.coordinate = algo_config.coordinate
       
        observation_dim = 4  # x, y, vel, yaw
        action_dim = 2  # acc, yawvel
        output_dim = 2  # acc, yawvel 
        

              
        vae_config = algo_config.vae             
        self.lstmvae = LSTMVAE(input_size=observation_dim+action_dim,
                           hidden_size=vae_config.hidden_size,#64
                           latent_size=vae_config.latent_size,
                           output_size=output_dim,
                           )
        self.default_chosen_inds = [0, 1, 2, 3, 4, 5] 
        
        diffuser_norm_info = algo_config.nusc_norm_info.diffuser
        norm_add_coeffs = diffuser_norm_info[0]
        norm_div_coeffs = diffuser_norm_info[1]
        self.add_coeffs = np.array(norm_add_coeffs).astype('float32')
        self.div_coeffs = np.array(norm_div_coeffs).astype('float32') 
              
        self.horizon = algo_config.horizon
        self.dt = 0.1
    
        self._dynamics_type = algo_config.dynamics.type
        self._dynamics_kwargs=algo_config.dynamics
        self._create_dynamics()
   
        self.context_encoder = ContextEncoder(observation_dim,
                                              algo_config,
                                              modality_shapes,
                                              self.dyn,
                                              )

    def _create_dynamics(self):
        if self._dynamics_type in ["Unicycle", dynamics.DynType.UNICYCLE]:
            self.dyn = dynamics.Unicycle(
                "dynamics",
                max_steer=self._dynamics_kwargs["max_steer"],
                max_yawvel=self._dynamics_kwargs["max_yawvel"],
                acce_bound=self._dynamics_kwargs["acce_bound"]
            )
        else:
            self.dyn = None

    def forward(self, batch,beta):
        aux_info,batch_state_and_action_scaled = self.pre_vae(batch)
        recon_act_output,mu,logvar = self.lstmvae(batch_state_and_action_scaled,aux_info["cond_feat"])
        recon_state_and_action_scaled = self.convert_action_to_state_and_action(recon_act_output,aux_info['curr_states'])
        recon_state_and_action_descaled = self.descale_traj(recon_state_and_action_scaled)
        loss,recon,kld = self.compute_vae_loss(batch_state_and_action_scaled,recon_state_and_action_descaled,mu,logvar,beta)
        return {"loss": loss, 
                'recon':recon,
                'kld':kld,
                'hist':batch['history_positions'],
                "input": batch['target_positions'],
                "output":recon_state_and_action_descaled[...,:2],
                "raster_from_agent":batch['raster_from_agent'],
                "image":batch['image'],
                }
        
    def pre_vae(self,batch):
        aux_info = self.context_encoder(batch)
        state_and_action = get_state_and_action_from_data_batch(batch)#[B,52,6]
        state_and_action_scaled = self.scale_traj(state_and_action)
        return aux_info,state_and_action_scaled

    def compute_vae_loss(self,input,output,mean,logvar,beta):
        
        recon = F.mse_loss(input,output)
        kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon + beta * kld
        return loss,recon,kld
    def convert_action_to_state_and_action(self, x_out, curr_states, scaled_input=True, descaled_output=False):
        '''
        Apply dynamics on input action trajectory to get state+action trajectory
        Input:
            x_out: (batch_size, num_steps, 2). scaled action trajectory
        Output:
            x_out: (batch_size, num_steps, 6). scaled state+action trajectory
        '''
        dim = len(x_out.shape)
        if dim == 4:
            B, N, T, _ = x_out.shape
            x_out = TensorUtils.join_dimensions(x_out,0,2)

        if scaled_input:
            x_out = self.descale_traj(x_out, [4, 5])
        x_out_state = unicyle_forward_dynamics(
            dyn_model=self.dyn,
            initial_states=curr_states,
            actions=x_out,
            step_time=self.dt,
            mode='parallel'
        )

        x_out_all = torch.cat([x_out_state, x_out], dim=-1)
        if scaled_input and not descaled_output:
            x_out_all = self.scale_traj(x_out_all, [0, 1, 2, 3, 4, 5])

        if dim == 4:
            x_out_all = x_out_all.reshape([B, N, T, -1])
        return x_out_all
  
    def scale_traj(self, target_traj_orig, chosen_inds=[]):
        '''
        - traj: B x T x D
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds

        squeeze_time_dim = False
        if target_traj_orig.dim() == 2:
        # 变成 [B, 1, D]
            target_traj_orig = target_traj_orig.unsqueeze(1)#[B,1,4]
            squeeze_time_dim = True


        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D     #[1,1,4]
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D     #[1,1,4]

        # TODO make these a buffer so they're put on the device automatically
        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device)
        target_traj = (target_traj_orig + dx_add) / dx_div
        if squeeze_time_dim:
            target_traj = target_traj.squeeze(1) 
        return target_traj 

    def descale_traj(self, target_traj_orig, chosen_inds=[]):
        '''
        - traj: B x T x D
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D

        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device) 

        target_traj = target_traj_orig * dx_div - dx_add
        

        return target_traj

   
        


   

