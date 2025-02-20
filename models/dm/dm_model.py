
import torch.nn as nn
import  torch
from tbsim.models.diffuser_helpers import (extract,cosine_beta_schedule)
from tbsim.utils.batch_utils import batch_utils
import tbsim.utils.tensor_utils as TensorUtils
import torch
import numpy as np
import tbsim.models.base_models as base_models
from tbsim.models.diffuser_helpers import unicyle_forward_dynamics,MapEncoder,convert_state_to_state_and_action
import torch.nn.functional as F
import tbsim.dynamics as dynamics
from tbsim.utils.diffuser_utils.progress import Progress, Silent
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
        
        self.register_buffer('x_t_cof',torch.sqrt(1. / alphas))
        self.register_buffer('noise_cof',betas / torch.sqrt(alphas - alphas_cumprod * alphas))

        
        
        self.model = TemporalMapUnet(horizon=algo_config.horizon,
                                      transition_dim=algo_config.vae.latent_size,
                                      cond_dim=algo_config.cond_feat_dim,
                                      output_dim=algo_config.vae.latent_size,
                                      dim=algo_config.base_dim,
                                      dim_mults=algo_config.dim_mults,
                                      diffuser_building_block='concat')
        self._dynamics_type = algo_config.dynamics.type
        self._dynamics_kwargs=algo_config.dynamics
        self._create_dynamics()
        '''
        diffuser_norm_info = algo_config.nusc_norm_info.diffuser
        norm_add_coeffs = diffuser_norm_info[0]
        norm_div_coeffs = diffuser_norm_info[1]
        self.add_coeffs = np.array(norm_add_coeffs).astype('float32')
        self.div_coeffs = np.array(norm_div_coeffs).astype('float32')
        '''
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

    def compute_losses(self,aux_info,z0):
        batch_size = len(z0)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=z0.device).long()# 128 在(0,100)之内
        noise_init = torch.randn_like(z0) #[B,52,4=latent_dim] 
        z_noisy = self.q_sample(x_0=z0, t=t, noise=noise_init) 
        noise = self.model(z_noisy, aux_info, t)#[B,52,latent]
        # z_recon = self.predict_start_from_noise(z_noisy, t=t, noise=noise)
        loss = F.mse_loss(noise_init,noise)
        return loss

    def q_sample(self, x_0, t, noise):        
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )
        return sample
    
    @torch.no_grad()
    def forward(self,data_batch,aux_info,algo_config):
        out_dict = self.sample_traj(data_batch,algo_config,aux_info)
 
        return out_dict
    def sample_traj(self,data_batch,algo_config,aux_info):
        batch_size = data_batch['history_positions'].size()[0]
        num_samp = algo_config.num_samp
        shape = (batch_size,algo_config.num_samp,algo_config.horizon,algo_config.vae.latent_size)#[B,N=1,52,4]
        #NOTE:p_sample_loop:      
        device = self.betas.device

        x = torch.randn(shape, device=device)#[B,N,52,4]
        x = TensorUtils.join_dimensions(x, begin_axis=0, end_axis=2)#[B*N,52,4]
        
        x_1 = None
        x_0 = None
        log_prob_final = None 
        aux_info = TensorUtils.repeat_by_expand_at(aux_info, repeats=num_samp, dim=0)

       
        steps = [i for i in reversed(range(0, self.n_timesteps, self.stride))]

        for i in steps:
            t = torch.full((batch_size*algo_config.num_samp,), i, device=device, dtype=torch.long)#[99,99,99,99...B*num_samp个]
            x, x_tminus1_mean, sigma = self.x_Tminus1(x,t,aux_info)
          
            
            if i==1:
                x_1 = x.clone()
            if i==0:
                x_0 = x.clone()
                dist = torch.distributions.Normal(x_tminus1_mean, sigma)
                log_prob_final = dist.log_prob(x)
                log_prob_final=log_prob_final.sum(dim=(1, 2))
                
        out_dict = {
                    'pred_traj' : x_0,#[B*N,52,4]
                    'x1':x_1, #[B,N,52,4]
                    'log_prob_final':log_prob_final, #[B*N]
                    'aux_info':aux_info
                    }
                

        return out_dict      
       
    def x_Tminus1(self,x,t,aux_info):
        b, *_, device = *x.shape, x.device

        noise_recon = self.model(x,aux_info,t)
        x_tminus1_mean,log_var = self.x_tminus1_mean_var(x,noise_recon,t)
        
        sigma = (0.5 * log_var).exp()
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        noise = torch.randn_like(x_tminus1_mean)
        noise = nonzero_mask * sigma * noise
        x_t_minus1 = x_tminus1_mean + noise
        return x_t_minus1, x_tminus1_mean, sigma

    def x_tminus1_mean_var(self,xt,noise,t):
        x_t_minus1_mean =  (
            extract(self.x_t_cof, t,xt.shape)*xt - extract(self.noise_cof,t,noise.shape)*noise
        )#[B*N,52,4]
        x_t_minus1_log_var = extract(self.posterior_log_variance_clipped, t, xt.shape)#[B*N,1,1]
        return x_t_minus1_mean,x_t_minus1_log_var
    
    def log_prob(self, x_t, x_t_minus_1, aux_info, t ):
        noise_recon = self.model(x_t,aux_info,t)
        x_tminus1_mean,log_var = self.x_tminus1_mean_var(x_t,noise_recon,t)
        sigma = (0.5 * log_var).exp()

        new_dist = torch.distributions.Normal(x_tminus1_mean,sigma)

        log_prob = new_dist.log_prob(x_t_minus_1)#[M*B*N,52,4]
        log_prob = log_prob.sum(dim=(1, 2))
        return log_prob
