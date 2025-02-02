
import torch.nn as nn
import  torch
from tbsim.models.diffuser_helpers import (extract,cosine_beta_schedule)

from .dm_mlp import MLPResNetwork
import tbsim.utils.tensor_utils as TensorUtils

from torch.distributions import Normal
import torch
import numpy as np
class DmModel(nn.Module):
    def __init__(
        self,
        latent_dim,
        cond_dim,
        time_dim,
        num_res_blocks,
        n_timesteps=100,
       
    ):
        super().__init__()
        self.n_timesteps = int(n_timesteps)
        self.stride = 1
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

        self.model = MLPResNetwork(latent_dim,cond_dim,time_dim,num_res_blocks)
        
    def compute_losses(self,z,aux_info):
        batch_size = len(z)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=z.device).long()# 128 在(0,1000)之内
        noise_init = torch.randn_like(z) #[B,128]
        x_start = z
        x_noisy = self.q_sample(x_start=x_start,t=t,noise=noise_init) #[B,128]
        
        noise_recon = self.model(x_noisy, aux_info, t)#[B,128]
        z_0_recon = self.predict_start_from_noise(x_noisy,t,noise_recon)
       
        return z_0_recon

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

   