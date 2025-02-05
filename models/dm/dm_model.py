
import torch.nn as nn
import  torch
from tbsim.models.diffuser_helpers import (extract,cosine_beta_schedule)
from tbsim.utils.batch_utils import batch_utils
import tbsim.utils.tensor_utils as TensorUtils
import torch
import numpy as np
import tbsim.models.base_models as base_models
from tbsim.models.diffuser_helpers import unicyle_forward_dynamics,MapEncoder,convert_state_to_state_and_action,AgentHistoryEncoder
import torch.nn.functional as F
import tbsim.dynamics as dynamics
from tbsim.models.temporal import TemporalMapUnet
class DmModel(nn.Module):
    def __init__(
        self,
        algo_config,
        modality_shapes,
        n_timesteps=100,
       
    ):
        super().__init__()
        self.n_timesteps = int(n_timesteps)
        self.stride = 1
        self.default_chosen_inds = [0, 1, 2, 3, 4, 5]
        self.horizon = algo_config.horizon
        self.dt = algo_config.step_time
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        cond_in_feat_size=0
        state_in_dim=4
        curr_state_feature_dim = algo_config.curr_state_feat_dim
        layer_dims = (curr_state_feature_dim, curr_state_feature_dim)
        self.agent_state_encoder = base_models.MLP(state_in_dim,
                                                       curr_state_feature_dim,
                                                       layer_dims,
                                                       normalization=True)
        
        cond_in_feat_size += curr_state_feature_dim

        self.map_encoder = MapEncoder(
                model_arch=algo_config.map_encoder_model_arch,
                input_image_shape=modality_shapes["image"],
                global_feature_dim=algo_config.map_feature_dim,
                grid_feature_dim= None,
            )
        
        cond_in_feat_size += algo_config.map_feature_dim #128+256=384
        
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
        
        self.model = TemporalMapUnet(horizon=algo_config.horizon,
                                      transition_dim=algo_config.transition_in_dim,
                                      cond_dim=cond_out_feat_size,
                                      output_dim=2,
                                      dim=algo_config.base_dim,
                                      dim_mults=algo_config.dim_mults,
                                      diffuser_building_block='concat')
        self._dynamics_type = algo_config.dynamics.type
        self._dynamics_kwargs=algo_config.dynamics
        self._create_dynamics()

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
    def compute_losses(self,data_batch):
        aux_info = self.get_aux_info(data_batch)
        future_traj = self.get_state_and_action_from_data_batch(data_batch)#(B,52,6)

        x = self.scale_traj(future_traj)
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()# 128 在(0,1000)之内
        noise_init = torch.randn_like(x) #[B,52,6]
      
        x_noisy = self.q_sample(x_start=x,t=t,noise=noise_init) #[B,128]
        x_action_noisy = x_noisy[..., [4, 5]]
        x_noisy = self.convert_action_to_state_and_action(x_action_noisy, aux_info['curr_states'])

        noise = self.model(x_noisy, aux_info, t)
        x_recon_action = self.predict_start_from_noise(x_action_noisy, t=t, noise=noise)
        x_recon = self.convert_action_to_state_and_action(x_recon_action, aux_info['curr_states'])

        x_recon= x_recon * data_batch['target_availabilities'][:, :self.horizon].unsqueeze(-1)
        x_start = x * data_batch['target_availabilities'][:, :self.horizon].unsqueeze(-1)
        loss = F.mse_loss(x_start,x_recon)
        future_recon = self.descale_traj(x_recon)
        return loss,future_traj,future_recon


    def get_aux_info(self,data_batch):
        N = data_batch["history_positions"].size(0)
        device = data_batch["history_positions"].device
        cond_feat_in = torch.empty((N,0)).to(device)
    

        curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.dyn.type())
        curr_state_feat = self.agent_state_encoder(curr_states)
        cond_feat_in = torch.cat([cond_feat_in, curr_state_feat], dim=-1)

        image_batch = data_batch["image"]
        map_global_feat, _ = self.map_encoder(image_batch)
        cond_feat_in = torch.cat([cond_feat_in, map_global_feat], dim=-1)

        cond_feat = self.process_cond_mlp(cond_feat_in)
        aux_info = {
            'cond_feat': cond_feat, 
            'curr_states': curr_states,
        }
        return aux_info





        '''
        hist_availability = data_batch['history_availabilities']#[B,31]

        hist_pos = data_batch['history_positions']#[B,31,2]
        hist_speed = data_batch['history_speeds']#[B,31]
        hist_yaw = data_batch['history_yaws']#[B,31,1]

        hist_pos[~hist_availability]=0.0
        hist_speed[~hist_availability] = 0.0 
        hist_yaw[~hist_availability] = 0.0
        hist_speed = hist_speed.unsqueeze(-1)#[B,31,1]

        hist_state = torch.cat([hist_pos, hist_speed, hist_yaw], dim=-1) #[B,31,4]
        hist_state = self.scale_traj(hist_state,[0,1,2,3])
        _, (h_n, _) = self.hist_encoder(hist_state)
        hist_features = h_n[-1]#[B,hid=128]
        agent_hist_feat = self.agent_hist_encoder(data_batch["history_positions"],
                                                      data_batch["history_yaws"],
                                                      data_batch["history_speeds"],
                                                      data_batch["extent"],
                                                      data_batch["history_availabilities"])
        curr_states = batch_utils().get_current_states(data_batch, dyn_type=self.dyn.type())#[B,4]
        image_batch = data_batch["image"]#[B,34,224,224]包含历史轨迹和邻居轨迹
        map_global_feat,_ = self.map_encoder(image_batch)#[B,256]
        cond_feat_in = torch.cat([hist_features, map_global_feat], dim=-1)#[B,128+256=384]
        cond_feat = self.process_cond_mlp(cond_feat_in)#[B,256]
        aux_info = {
            'cond_feat':cond_feat,
            'curr_states':curr_states
        }
        return aux_info
        '''
        
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
    def q_sample(self, x_start, t, noise):        
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample
   
    def predict_start_from_noise(self, x_t, t, noise):

        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
  
    def forward(self,batch,aux_info,algo_config):
        batch_size = batch['history_positions'].size()[0]
        shape = (batch_size,algo_config.num_samp,algo_config.vae.latent_size)#[B,N=1,128]
        #NOTE:p_sample_loop:
        
        device = self.betas.device
        x = torch.randn(shape, device=device)#[B,N,128]
        x = TensorUtils.join_dimensions(x, begin_axis=0, end_axis=2)#[B*N,128]
        noise_all = torch.randn(
            x.shape[0],  # B*N
            self.n_timesteps,  # total steps
            algo_config.vae.latent_size,  # D
            device=device
        )#[B,100,128]

        aux_info = TensorUtils.repeat_by_expand_at(aux_info, repeats=algo_config.num_samp, dim=0)

        steps = [i for i in reversed(range(0, self.n_timesteps, self.stride))]
        log_probs = []
        traj_data = []
        for i in steps:
            timesteps = torch.full((x.shape[0],), i, device=device, dtype=torch.long)#[99,99,99,99...B个]
            noise_t = noise_all[:, i, :]#[B,128]
            x_tminus1, mean_t, sigma_t  = self.x_minus1(x,timesteps,noise_t,aux_info)

            step_info = {
            "x_t": x,
            "x_tminus1": x_tminus1,
            "mean_t": mean_t,
            "sigma_t": sigma_t,
            "t":timesteps
        }
            traj_data.append(step_info)
            # log_probs.append(log_prob_step)

            x = x_tminus1 
       
        # #TODO:添加对于静止车辆的过滤
        
        
        return x, traj_data
       
    def x_minus1(self,x,t,noise,aux_info):
        b = x.shape[0]
        model_mean, posterior_variance,model_log_variance = self.x_tminus1_mean(x=x, t=t,aux_info=aux_info)
        sigma = (0.5 * model_log_variance).exp()
        
        
        nonzero_mask = (1 - (t == 0).float()).reshape(b , *((1,) * (len(x.shape) - 1)))
        
        
        noise = nonzero_mask * sigma * noise
        x_tminus1 = model_mean + noise
        # dist = Normal(model_mean,sigma)
        # log_prob_step = dist.log_prob(x_tminus1).sum(dim=-1)
        return x_tminus1,model_mean,sigma
   
    def x_tminus1_mean(self,x,t,aux_info):
        noise_recon = self.model(x, aux_info, t)#[B,128]
        x_0_recon = self.predict_start_from_noise(x, t=t, noise=noise_recon)#[B,128]
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_0_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

   
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

        if scaled_input: #x_out 动作 维度=2
            x_out = self.descale_traj(x_out, [4, 5])#(B,52,2)
        x_out_state = unicyle_forward_dynamics(
            dyn_model=self.dyn,
            initial_states=curr_states,#(B,4)
            actions=x_out,
            step_time=self.dt,
            mode='chain'
        )

        x_out_all = torch.cat([x_out_state, x_out], dim=-1)
        if scaled_input and not descaled_output:
            x_out_all = self.scale_traj(x_out_all, [0, 1, 2, 3, 4, 5])

        if dim == 4:
            x_out_all = x_out_all.reshape([B, N, T, -1])
        return x_out_all