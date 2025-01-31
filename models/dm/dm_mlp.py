import torch.nn as nn
from tbsim.models.diffuser_helpers import SinusoidalPosEmb
import torch

class ResidualBlock(nn.Module):

    def __init__(self, x_dim, cond_dim):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, cond_dim)
        self.act = nn.Mish()
        self.fc2 = nn.Linear(cond_dim, x_dim)

    def forward(self, x, cond):
        out = self.fc1(x)+cond
        out = self.act(out)
        out = self.fc2(out)
        return out
class MLPResNetwork(nn.Module):
    def __init__(
            self,
            latent_dim,#128
            cond_dim, #256
            time_dim, #128
            num_res_blocks
            ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(
                x_dim=latent_dim,
                cond_dim=(cond_dim + time_dim),
            )
            for _ in range(num_res_blocks)
        ])
    def forward(self,x,aux_info,time):

        t_emb = self.time_mlp(time) # [B, time_dim=128]
        context = aux_info['context']#[B,256]
        cond = torch.cat([t_emb, context], dim=-1)#[B,128+256]

        for block in self.blocks:
            x = block(x, cond)

        
        return x
       
