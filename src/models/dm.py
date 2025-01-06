from tbsim.models.diffuser import DiffuserModel
import torch.nn as nn
import torch.nn.functional as F
import  torch
from tbsim.models.diffuser_helpers import (extract,cosine_beta_schedule)
from .dm_mlp import MLPResNetwork
class DM(nn.Module):
    def __init__(
        self,
        latent_dim,
        cond_dim,
        n_timesteps=1000,
       
    ):
        super().__init__()
        self.n_timesteps = int(n_timesteps)
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

         
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        self.model = MLPResNetwork(latent_dim = latent_dim,cond_dim = cond_dim)
        
    def compute_losses(self,z,aux_info):
        batch_size = len(z)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=z.device).long()
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
   
    def predict_start_from_noise(self, x_t, t, noise, force_noise=False):

        pass