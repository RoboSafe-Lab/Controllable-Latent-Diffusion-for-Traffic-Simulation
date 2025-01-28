
import torch.nn as nn
import torch.nn.functional as F
import  torch
from tbsim.models.diffuser_helpers import (extract,cosine_beta_schedule)

from .dm_mlp import MLPResNetwork
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.diffuser_utils.progress import Progress,Silent
from torch.distributions import Normal
import torch
import numpy as np
class DmModel(nn.Module):
    def __init__(
        self,
        latent_dim,
        cond_dim,
        time_dim,
        hidden_dim,
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
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.model = MLPResNetwork(latent_dim,cond_dim,time_dim,hidden_dim,num_res_blocks)
        
    def compute_losses(self,z,aux_info):
        batch_size = len(z)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=z.device).long()# 128 在(0,1000)之内
        noise_init = torch.randn_like(z) #[B,128]
        x_start = z
        x_noisy = self.q_sample(x_start=x_start,t=t,noise=noise_init) #[B,128]
        t_inp = t
        noise_recon = self.model(x_noisy, aux_info, t_inp)#[B,128]
        loss = F.mse_loss(noise_recon,noise_init)
        return loss

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
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def forward(self,batch,aux_info,algo_config):
        batch_size = batch['history_positions'].size()[0]
        shape = (batch_size,algo_config.num_samp,algo_config.vae.latent_size)#[B,N,128]
        return self.p_sample_loop(shape,batch,algo_config,aux_info)

    def p_sample_loop(self,shape,batch,algo_config,aux_info):
        #NOTE:p_sample_loop:
        batch_size = shape[0]
        device = self.betas.device
        x = torch.randn(shape, device=device)#[B,N,128]
        x = TensorUtils.join_dimensions(x, begin_axis=0, end_axis=2)#[B*N,128]
    
        num_samp = algo_config.num_samp
        aux_info = TensorUtils.repeat_by_expand_at(aux_info, repeats=num_samp, dim=0)

        steps = [i for i in reversed(range(0, self.n_timesteps, self.stride))]
        for i in steps:
            timesteps = torch.full((batch_size*num_samp,), i, device=device, dtype=torch.long)
            x, guide_losses = self.p_sample(x,timesteps,batch,algo_config,aux_info)
            


        z_0 = TensorUtils.reshape_dimensions(x, begin_axis=0, end_axis=1, target_dims=(batch_size, num_samp))
        #TODO:添加对于静止车辆的过滤
        
        log_prob = ...
        return z_0,log_prob
       
    def p_sample(self,x,t,batch,algo_config,aux_info):
        b, *_, device = *x.shape, x.device
        model_mean, posterior_variance, model_log_variance, q_posterior_in = self.p_mean_variance(x=x, t=t,aux_info=aux_info)
        sigma = (0.5 * model_log_variance).exp()
        x_initial = model_mean.clone().detach()
        return_grad_of = x_initial
        nonzero_mask = (1 - (t == 0).float()).reshape(b , *((1,) * (len(x.shape) - 1)))
        x_initial.requires_grad_()
        noise = torch.randn_like(x_initial)
        noise = nonzero_mask * sigma * noise
        x_out = x_initial + noise
        return x_out
   

    def p_mean_variance(self,x,t,aux_info):
        t_inp = t
        x_model_in = x
        noise_recon = self.model(x_model_in, aux_info, t_inp)

        x_tmp = x.detach()
        x_0_recon = self.predict_start_from_noise(x_tmp, t=t, noise=noise_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_0_recon, x_t=x_tmp, t=t)

        return model_mean, posterior_variance, posterior_log_variance, (x_0_recon, x_tmp, t)
    
    def compute_log_q(self, x_prev, x_t, x_0_recon, t):
    # same code snippet
        model_mean, posterior_variance, _ = self.q_posterior(x_0_recon, x_t, t)
        std = torch.sqrt(posterior_variance)
        dist = torch.distributions.Normal(model_mean, std)
        # if shape is [B, D], we do:
        log_q_per_dim = dist.log_prob(x_prev)   # shape [B, D]
        log_q = log_q_per_dim.sum(dim=-1)       # shape [B]
        return log_q