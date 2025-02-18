import torch.nn as nn
import torch
import torch.nn.functional as F
from l5kit.geometry import transform_points
from collections import deque
import random
def compute_reward(trajectory,batch):
    B, N, T, D = trajectory.shape
    raster_from_agent = batch['raster_from_agent']#[B,3,3]

    traj_raster = transform_points_tensor(trajectory, raster_from_agent)#[B,N,52,2]    

    drivable_map = batch['drivable_map']#[B,224,224]

    traj_int = traj_raster.round().long()
    cols = traj_int[..., 0].clamp(0, drivable_map.shape[-1] - 1)
    rows = traj_int[..., 1].clamp(0, drivable_map.shape[-2] - 1)

    batch_idx = torch.arange(B,device=drivable_map.device).view(B,1,1).expand(B, N, T)

    traj_values = drivable_map[batch_idx,rows,cols]#[B,_num_samp,52]

    reward_per_timestep = torch.where(traj_values == False,torch.tensor(-1.0,device=drivable_map.device),
                                      torch.tensor(0.0,device=drivable_map.device))

    offroad_reward = reward_per_timestep.sum(dim=-1)#[B,5]
    # offroad_reward_mean = offroad_reward.mean(dim=1)

    collision_reward = compute_collision_reward(trajectory,batch)
    # collision_reward_mean =  collision_reward.mean(dim=1)

    return offroad_reward+collision_reward
    
def compute_collision_reward(traj,batch,collision_thresh=0.2):
    B, N, T, _ = traj.shape
    ego_pos = traj

    other_pos = batch['all_other_agents_future_positions']#[B,73,52,2]
    avail = batch['all_other_agents_future_availability'] #[B,75,T]

    traj_exp = traj.unsqueeze(2)#[B,N,1,T,2]
    other_pos_exp = other_pos.unsqueeze(1)#[B,1,A,T,2]

    diff = traj_exp - other_pos_exp #[B,N,A,T,2]
    distance = torch.norm(diff, dim=-1) #[B,N,A,T]

    collision_mask = distance < collision_thresh

    avail_exp = avail.unsqueeze(1) #[B,1,A,T]
    valid_collision = collision_mask & avail_exp

    collision_occur = valid_collision.any(dim=2)#[B,N,T]

    collision_reward_per_timestep = torch.where(
        collision_occur,
        torch.tensor(-1.0, device=traj.device),
        torch.tensor(0.0, device=traj.device)
    )#[B,N,52]


    collision_reward = collision_reward_per_timestep.sum(dim=-1)#[B,N]
    
    return collision_reward

def transform_points_tensor(trajectory_position,raster_from_agent):
    B, N, T, F = trajectory_position.shape
    points = trajectory_position.reshape(B, N * T, F)
    num_dims = raster_from_agent.shape[-1] - 1 
    T_matrix = raster_from_agent.transpose(1, 2)

    linear_part = T_matrix[:, :num_dims, :num_dims]#[B,2,2]
    translation_part = T_matrix[:, -1:, :num_dims]#[B,1,2]

    transformed_points = torch.bmm(points, linear_part) + translation_part
    transformed_trajectory = transformed_points.reshape(B, N, T, F)
    return transformed_trajectory



class ReplayBuffer:
    def __init__(self,capacity=10):
        self.capacity = capacity
        self.buffer = deque(maxlen = capacity)
    
    def add(self,x0,x1,log_prob_old,advantage,aux_info):
        self.buffer.append((
            x0.detach().cpu(),
            x1.detach().cpu(), 
            log_prob_old.detach().cpu(), 
            advantage.detach().cpu(),
            aux_info.detach().cpu(),
            ))

    def sample(self,batch_size):
        return random.sample(self.buffer,batch_size)
    
    def clear(self):
        self.buffer.clear()
    def __len__(self):
        return len(self.buffer)
        


def detach_aux_info(aux_info):
    detached = {}
    for key, value in aux_info.items():
        if torch.is_tensor(value):
            detached[key] = value.detach().cpu()
        else:
            detached[key] = value  # 如果不是 tensor，直接保存
    return detached










'''

    import matplotlib.pyplot as plt

    fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(16,16))
    i1=1
    i2=2
    image = batch['image'].permute(0,2,3,1).detach().cpu().numpy()
    image1 = image[i1][...,-3:]*0.5+0.5
    image2 = image[i2][...,-3:]*0.5+0.5
    image3 = image[3][...,-3:]*0.5+0.5
    image4 = image[4][...,-3:]*0.5+0.5

    ax1.imshow(image1)
    ax2.imshow(image2)
    ax3.imshow(image3)
    ax4.imshow(image4)

    traj_raster=traj_raster.detach().cpu().numpy()
    for i in range(N):
        ax1.scatter(traj_raster[i1,i,:,0],traj_raster[i1,i,:,1],c='b',s=0.2)
        ax2.scatter(traj_raster[i2,i,:,0],traj_raster[i2,i,:,1],c='r',s=0.2)
        ax3.scatter(traj_raster[3,i,:,0],traj_raster[3,i,:,1],c='b',s=0.2)
        ax4.scatter(traj_raster[4,i,:,0],traj_raster[4,i,:,1],c='r',s=0.2)
    '''