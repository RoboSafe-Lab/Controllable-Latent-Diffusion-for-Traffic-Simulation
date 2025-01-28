
import torch.nn as nn
import numpy as np
from models.vae.lstm_vae import LSTMVAE
import torch
from tbsim.utils.batch_utils import batch_utils
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.dynamics as dynamics
import tbsim.models.base_models as base_models
from tbsim.models.diffuser_helpers import MapEncoder,convert_state_to_state_and_action,unicyle_forward_dynamics

class VaeModel(nn.Module):
    def __init__(self, algo_config,train_config, modality_shapes):

        super(VaeModel, self).__init__()
        self.algo_config = algo_config
        self.train_config = train_config
        self.data_centric = None
        self.coordinate = algo_config.coordinate
        print(f"algo_config_diffuser_input_mode: {algo_config.diffuser_input_mode}")
     
       
        observation_dim = 4  # x, y, vel, yaw
        action_dim = 2  # acc, yawvel
        output_dim = 2  # acc, yawvel   
        layer_dims =  (algo_config.curr_state_feat_dim, algo_config.curr_state_feat_dim)  #(64,64)
        self.default_chosen_inds = [0, 1, 2, 3, 4, 5] 
        
        cond_in_feat_size = 0
        cond_in_feat_size += algo_config.curr_state_feat_dim #64
        self.agent_state_encoder = base_models.MLP(observation_dim,
                                                       algo_config.curr_state_feat_dim,
                                                       layer_dims,
                                                       normalization=True)
        self.map_encoder = MapEncoder(
                model_arch=algo_config.map_encoder_model_arch,
                input_image_shape=modality_shapes["image"],
                global_feature_dim=algo_config.map_feature_dim,
                grid_feature_dim= None,
            )
        cond_in_feat_size += algo_config.map_feature_dim #64+256=320
        cond_out_feat_size = algo_config.cond_feat_dim #256
        combine_layer_dims = (cond_in_feat_size, cond_in_feat_size, cond_out_feat_size, cond_out_feat_size)
        self.process_cond_mlp = base_models.MLP(cond_in_feat_size,
                                                cond_out_feat_size,
                                                combine_layer_dims,
                                                normalization=True)
        

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
   
        vae_config = algo_config.vae             
        self.lstmvae = LSTMVAE(input_size=observation_dim+action_dim,
                           hidden_size=vae_config.hidden_size,
                           latent_size=vae_config.latent_size,
                           output_size=output_dim,
                           )
     


        
       

        '''
        (B,256)->(B,1,256)

        (B,52,x,y,vel,yaw,)
            +
        map:(B,seq,256)考虑current_state 目前先这样做,以后去掉也方便
        '''
        # set up EMA


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
        aux_info,unscaled_input,scaled_input = self.pre_vae(batch)
        scaled_actions,mu,logvar = self.lstmvae(scaled_input,aux_info["cond_feat"])
        scaled_output = self.convert_action_to_state_and_action(scaled_actions,aux_info['curr_states'])

        descaled_output = self.descale_traj(scaled_output)
        losses = self.lstmvae.loss_function(scaled_output,scaled_input,mu,logvar,beta)
        return {"loss": losses['loss'], 
                "input": unscaled_input,
                "output":descaled_output,
                "raster_from_agent":batch['raster_from_agent'],
                "maps":batch['maps'],
                }, losses
        
    def z2traj(self,z,num_samp,aux_info):
        scaled_actions,mu,logvar = self.lstmvae.getTraj(z,num_samp)
        scaled_output = self.convert_action_to_state_and_action(scaled_actions,aux_info['curr_states'])
        return scaled_output
  



      
     
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
    def pre_vae(self,batch):
        aux_info = self.get_aux_info(batch)


        unscaled_input = self.get_state_and_action_from_data_batch(batch)#[B,52,6]
        scaled_input = self.scale_traj(unscaled_input)
        return aux_info,unscaled_input,scaled_input

    def get_aux_info(self,data_batch):
        N = data_batch["history_positions"].size(0)
        device = data_batch["history_positions"].device
        cond_feat_in = torch.empty((N,0)).to(device)

        curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.dyn.type())#[B,4]
        curr_states_input = self.scale_traj(curr_states,[0,1,2,3])#[B,4]
        curr_state_feat = self.agent_state_encoder(curr_states_input)#[B,64]
        cond_feat_in = torch.cat([cond_feat_in, curr_state_feat], dim=-1)#[B,0+64]

        image_batch = data_batch["image"]#[B,34,224,224]包含历史轨迹和邻居轨迹
        map_global_feat, map_grid_feat = self.map_encoder(image_batch)#[B,256]
        cond_feat_in = torch.cat([cond_feat_in, map_global_feat], dim=-1)#[B,64+256=320]
        cond_feat = self.process_cond_mlp(cond_feat_in)#[B,256]
        aux_info = {
            'cond_feat': cond_feat, #已经归一化输入
            'curr_states': curr_states, #没有归一化
        }
        return aux_info
    
    def get_state_and_action_from_data_batch(self, data_batch, chosen_inds=[]):
        '''
        Extract state and(or) action from the data_batch from data_batch
        Input:
            data_batch: dict
        Output:
            x: (batch_size, num_steps, len(chosen_inds)).
        '''
        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        # NOTE: for predicted agent, history and future with always be fully available
        traj_state = torch.cat(
                (data_batch["target_positions"][:, :self.horizon, :], data_batch["target_yaws"][:, :self.horizon, :]), dim=2)

        traj_state_and_action = convert_state_to_state_and_action(traj_state, data_batch["curr_speed"], self.dt)

        return traj_state_and_action[..., chosen_inds]
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

   
        


   

