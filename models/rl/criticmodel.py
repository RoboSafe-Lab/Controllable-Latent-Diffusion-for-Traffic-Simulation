import torch.nn as nn
import torch
import torch.nn.functional as F
from l5kit.geometry import transform_points
class CriticModel(nn.Module):
    def __init__(self,state_dim, hidden_dim,lr,collision_penalty):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.collision_penalty = collision_penalty
    def forward(self, batch,trajectory):

        reward = self.compute_reward(batch,trajectory)
    
    def train_critic(self, states, returns):
   
        self.optimizer.zero_grad()
        pred_values = self(states)  
        
        loss = F.mse_loss(pred_values, returns)
        
        loss.backward()
        self.optimizer.step()

        return loss
    def compute_reward(self, batch,trajectory):

        raster_from_agent = batch['raster_from_agent']

        drivable_map = batch['drivable_map']
        raster_traj = transform_points(trajectory,raster_from_agent)
        is_on_drivable = drivable_map[:, raster_traj[..., 1], raster_traj[..., 0]]
        out_of_bounds_penalty = (is_on_drivable == 0).float().sum(dim=-1)

        lane_points = batch['extras']['closest_lane_point'][..., :2]
        trajectory_expanded = trajectory[:, :, None, None, :]  # [B, T, 1, 1, 2]

        # lane_distances = torch.norm(trajectory_expanded - lane_points[:, None, :, :, :], dim=-1)  # [B, T, S_seg, S_point]
        # # min_lane_distance, _ = torch.min(lane_distances.view(4, 52, -1), dim=-1)  # [B, T]

        # lane_alignment_penalty = min_lane_distance.mean(dim=-1)  # [B]




