
import torch.nn as nn
import torch.nn.functional as F
import  torch
from tbsim.models.diffuser_helpers import (extract,cosine_beta_schedule)

from .dm_mlp import MLPResNetwork
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.diffuser_utils.progress import Progress,Silent

import numpy as np
class DmModel(nn.Module):
    def __init__(
        self,
        latent_dim,
        cond_dim,
        time_dim,
        hidden_dim,
        num_res_blocks,
        n_timesteps=1000,
       
    ):
        super().__init__()
        self.n_timesteps = int(n_timesteps)
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

    def forward(self,z,aux_info,num_samp=1):

       
        batch_size,latent = z.shape
        shape = (batch_size, num_samp, latent)
        device = self.betas.device
        x = torch.randn(shape, device=device) # (B, N, D)
        x = TensorUtils.join_dimensions(x, begin_axis=0, end_axis=2) # B*N, D
        aux_info = TensorUtils.repeat_by_expand_at(aux_info, repeats=num_samp, dim=0)
        bN = x.shape[0]
        progress = Progress(self.n_timesteps)
        steps = [i for i in reversed(range(0, self.n_timesteps, 1))] #1000
        
        for i in steps:
            progress.update({'t': i})
            timesteps = torch.full((batch_size*num_samp,), i, device=device, dtype=torch.long)
            with torch.no_grad():
                t_inp = timesteps
                model_input = self.model(x,aux_info,t_inp)
                x_tmp = x.detach()
                x_recon = (
                extract(self.sqrt_recip_alphas_cumprod, timesteps, x_tmp.shape) * x_tmp -
                extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_tmp.shape) * model_input
            )
                
                model_mean = (
            extract(self.posterior_mean_coef1, timesteps, x_tmp.shape) * x_recon +
            extract(self.posterior_mean_coef2, timesteps, x_tmp.shape) * x_tmp
        )       
                posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, timesteps, x_tmp.shape)
            sigma = (0.5 * posterior_log_variance_clipped).exp()

            # nonzero_mask = (1 - (timesteps == 0).float()).reshape(batch_size* num_samp, *((1,) * (len(x.shape) - 1)))
            nonzero_mask = (1 - (timesteps == 0).float()).reshape(bN, 1)
            x_initial = model_mean.clone().detach()
            noise = torch.randn_like(x_initial)
            noise = nonzero_mask * sigma * noise
            x_out = x_initial + noise

            x = x_out

        return x_out