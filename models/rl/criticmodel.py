import torch.nn as nn
import torch
import torch.nn.functional as F

class CriticModel(nn.Module):
    def __init__(self,state_dim, hidden_dim,lr,collision_penalty):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出一个标量Value
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.collision_penalty = collision_penalty
    def forward(self, state):
        return self.net(state)
    
    def train_critic(self, states, returns):
   
        self.optimizer.zero_grad()
        pred_values = self(states)  
        
        loss = F.mse_loss(pred_values, returns)
        
        loss.backward()
        self.optimizer.step()

        return loss
    def compute_reward(self, trajectory):

        B, T, D = trajectory.shape
        device = trajectory.device
        
        # 1) 简易超速惩罚
        speeds = trajectory[..., 2]     # [B, T]
        over_speed_mask = (speeds > self.speed_limit)
        # 这里例子: 每次超速时刻记 -1 => summation
        speed_penalty = over_speed_mask.sum(dim=1).float()  # shape [B], 超速多少次
        
        # 2) 碰撞惩罚(示例)
        #   假设 "detect_collision" 是自己写的函数, 
        #   这里我们就简单地把距离其它车<1视为碰撞 => -5 
        #   如果你没有其它车信息,就跳过
        collision_mask = self.detect_collision(trajectory)  # [B] bool
        # 只要有一次碰撞 -> self.collision_penalty
        collision_penalty_tensor = collision_mask.float() * self.collision_penalty  # shape [B]

        # 3) 最终回报(整条轨迹的累计):
        returns = -speed_penalty + collision_penalty_tensor
        # 也可以加其它逻辑: 例如到达目标 => +10, etc.

        return returns