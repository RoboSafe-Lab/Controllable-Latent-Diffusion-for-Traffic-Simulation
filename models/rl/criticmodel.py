
import torch


from collections import deque
import random
def compute_reward(state_act,batch,state_act_scaled):
    with torch.no_grad():
        trajectory = state_act[...,:2]
        B, N, T, D = trajectory.shape
        raster_from_agent = batch['raster_from_agent']

        traj_raster = transform_points_tensor(trajectory, raster_from_agent)   

        drivable_map = batch['drivable_map']

        traj_int = traj_raster.round().long()
        cols = traj_int[..., 0].clamp(0, drivable_map.shape[-1] - 1)
        rows = traj_int[..., 1].clamp(0, drivable_map.shape[-2] - 1)

        batch_idx = torch.arange(B,device=drivable_map.device).view(B,1,1).expand(B, N, T)

        traj_values = drivable_map[batch_idx,rows,cols]   


        reward_per_timestep = torch.where(traj_values == False,torch.tensor(-1.0,device=drivable_map.device),
                                        torch.tensor(0.0,device=drivable_map.device))

        offroad_reward = reward_per_timestep.sum(dim=-1)
       

        collision_reward = compute_collision_reward(trajectory,batch)

        dt = 0.1
        acc = state_act_scaled[...,4]
        jerk = (acc[:, :, 1:] - acc[:, :, :-1]) / dt
        jerk_penalty = jerk.abs().mean(dim=-1)
        reward = (offroad_reward+collision_reward-jerk_penalty*0.1).view(-1)

        return reward
    
def compute_collision_reward(traj,batch,collision_thresh=0.8):
    with torch.no_grad():
        if traj.dim() == 3:
            traj = traj.unsqueeze(1)
            B, N, T, D = traj.shape
            other_pos = batch['all_other_agents_future_positions']
            avail = batch['all_other_agents_future_availability']

            T_other = other_pos.size(2) 
            if T != T_other:
                traj = traj[..., :T_other, :] 
                T = T_other

            traj_exp = traj.unsqueeze(2)
            other_pos_exp = other_pos.unsqueeze(1)
            diff = traj_exp - other_pos_exp
            distance = torch.norm(diff, dim=-1)
            collision_mask = distance < collision_thresh
            avail_exp = avail.unsqueeze(1)
            valid_collision = collision_mask & avail_exp
            collision_count = valid_collision.float().sum(dim=(2, 3))
            collision_reward = -collision_count
            return collision_reward
        # B, N, T, _ = traj.shape
        
        # ego_pos = traj

        # other_pos = batch['all_other_agents_future_positions']#[B,S,52,2]
        # avail = batch['all_other_agents_future_availability'] #[B,S,52] S个其他agents

        # traj_exp = traj.unsqueeze(2)#[B,N,1,52,2]
        # other_pos_exp = other_pos.unsqueeze(1)#[B,1,S,52,2]

        # diff = traj_exp - other_pos_exp #[B,N,S,T,2]
        # distance = torch.norm(diff, dim=-1) #[B,N,S,52]

        # collision_mask = distance < collision_thresh

        # avail_exp = avail.unsqueeze(1) #[B,1,S,52]
        # valid_collision = collision_mask & avail_exp#[B,N,S,52]

        # collision_count = valid_collision.float().sum(dim=2)# [B, N, 52]
        # collision_reward = -collision_count.sum(dim=-1) # [B, N]

        # return collision_reward

def transform_points_tensor(trajectory_position,raster_from_agent):
    # with torch.no_grad():
    #     B, N, T, F = trajectory_position.shape
    #     points = trajectory_position.reshape(B, N * T, F)
    #     num_dims = raster_from_agent.shape[-1] - 1 
    #     T_matrix = raster_from_agent.transpose(1, 2)

    #     linear_part = T_matrix[:, :num_dims, :num_dims]#[B,2,2]
    #     translation_part = T_matrix[:, -1:, :num_dims]#[B,1,2]

    #     transformed_points = torch.bmm(points, linear_part) + translation_part
    #     transformed_trajectory = transformed_points.reshape(B, N, T, F)
    #     return transformed_trajectory
    with torch.no_grad():
        B,  T, F = trajectory_position.shape
        points = trajectory_position.reshape(B, T, F)
        num_dims = raster_from_agent.shape[-1] - 1 
        T_matrix = raster_from_agent.transpose(1, 2)

        linear_part = T_matrix[:, :num_dims, :num_dims]#[B,2,2]
        translation_part = T_matrix[:, -1:, :num_dims]#[B,1,2]

        transformed_points = torch.bmm(points, linear_part) + translation_part
        transformed_trajectory = transformed_points.reshape(B, T, F)
        return transformed_trajectory

def failure_rate_compute(state_action, batch):
    with torch.no_grad():
        trajectory = state_action[...,:2]
        B, T, D = trajectory.shape
        raster_from_agent = batch['raster_from_agent']

        traj_raster = transform_points_tensor(trajectory, raster_from_agent)  
        drivable_map = batch['drivable_map']  
        traj_int = traj_raster.round().long()
        cols = traj_int[..., 0].clamp(0, drivable_map.shape[-1] - 1)  
        rows = traj_int[..., 1].clamp(0, drivable_map.shape[-2] - 1)
        batch_idx = torch.arange(B, device=drivable_map.device).view(B,1).expand(B,  T)
        traj_values = drivable_map[batch_idx, rows, cols]  
        
        offroad_flag = (traj_values != False)  
        no_offroad_rate = offroad_flag.all(dim=-1).float().mean().item()  
        
        collision_reward = compute_collision_reward(trajectory, batch)  
        no_collision_rate = (collision_reward >= 0).float().mean().item()

       
        offroad_failure_rate = 1.0 - no_offroad_rate
        collision_failure_rate = 1.0 - no_collision_rate

    
        overall_failure_rate = (offroad_failure_rate + collision_failure_rate) / 2.0

        return {
            'offroad_failure_rate': offroad_failure_rate,
            'collision_failure_rate': collision_failure_rate,
            'overall_failure_rate': overall_failure_rate
        }

class ReplayBuffer:
    def __init__(self,capacity=10000, alpha = 0.9):
        self.capacity = capacity
        self.buffer = deque(maxlen = capacity)

        self.running_reward_baseline = 0.0
        self.has_init_baseline = False
        self.alpha = alpha
    def add(self,x0,x1,log_p_old,reward,cond_feat_value):
        current_batch_mean_r = reward.mean().item()
        if not self.has_init_baseline:
            self.running_reward_baseline = current_batch_mean_r
            self.has_init_baseline = True
        else:
            self.running_reward_baseline = (self.alpha*self.running_reward_baseline + (1-self.alpha)*current_batch_mean_r)
        batch_size = x0.shape[0]
        for i in range(batch_size):

            sample_x0 = x0[i].detach().cpu()
            sample_x1 = x1[i].detach().cpu()
            sample_log_p = log_p_old[i].detach().cpu()
            sample_reward = reward[i].detach().cpu()
            sample_cond_feat = cond_feat_value[i]
 
            self.buffer.append((
                sample_x0,
                sample_x1,
                sample_log_p,
                sample_reward,
                sample_cond_feat,
                ))
    def get_baseline(self):
        return self.running_reward_baseline
    
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
            detached[key] = value 
    return detached







