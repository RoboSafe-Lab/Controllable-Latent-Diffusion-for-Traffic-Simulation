import torch.nn as nn
from tbsim.models.diffuser_helpers import SinusoidalPosEmb
import torch

class ResidualBlockMLP(nn.Module):
 
    def __init__(self, dim=256, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim  
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.net(x) 


class MLPResNetwork(nn.Module):
    def __init__(
            self,
            latent_dim,
            cond_dim, 
            time_dim, 
            hidden_dim,
            num_res_blocks
            ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.initial_layer = nn.Sequential(
            nn.Linear(latent_dim + time_dim + cond_dim, hidden_dim),
            nn.Mish()
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlockMLP(hidden_dim, hidden_dim) for _ in range(num_res_blocks)
        ])
        self.final_layer = nn.Linear(hidden_dim, latent_dim)
    def forward(self,x,aux_info,time):

        t_emb = self.time_mlp(time) # [B, time_dim]
        cond_feat = aux_info['cond_feat']#[B,256]
        combined = torch.cat([x, t_emb, cond_feat], dim=-1)
        h = self.initial_layer(combined)

        for block in self.res_blocks:
            h = block(h) 


        out = self.final_layer(h)
        return out
