import torch.nn as nn
import torch
import torch.nn.functional as F


def compute_reward(batch, trajectory, aux_info):
    raster_from_agent = batch['raster_from_agent']

    drivable_map = batch['drivable_map']
    trajectory_position = trajectory[...,:2]
    traj_raster = transform_points_torch(trajectory_position,raster_from_agent)
    offroad_reward = compute_offroad_reward_torch(traj_raster, drivable_map, threshold=0.5)
    collision_reward = compute_collision_reward(trajectory_position,batch)
    return None
    
def compute_collision_reward(traj,batch,collision_thresh=0.2):
    B, N, T, _ = batch['all_other_agents_future_positions'].shape 
    valid_positions = batch['all_other_agents_future_positions'].clone()
    valid_positions[batch['all_other_agents_future_availability'] == 0] = 1e6

    traj_exp = traj.unsqueeze(1)
    distances = torch.norm(valid_positions - traj_exp, dim=-1)
    collision_mask = distances < collision_thresh 
    collision_per_sample = collision_mask.any(dim=1).any(dim=1)
    collision_reward = torch.where(collision_per_sample, torch.tensor(-1.0, device=traj.device), torch.tensor(1.0, device=traj.device))
    return collision_reward

def compute_offroad_reward_torch(traj_raster: torch.Tensor, drivable_map: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    根据轨迹在像素坐标下的位置以及 drivable_map 判断是否 offroad。
    
    参数:
      traj_raster: [B, T, 2] 的 tensor,表示轨迹在像素坐标系下的位置（x, y）。
      drivable_map: 可行区域图，可以是形状 [H, W]（全局同一张地图）或 [B, H, W]（每个样本独有）。
      threshold: 阈值，drivable_map 中小于此值认为不可行。
      
    返回:
      reward: 形状 [B] 的 tensor，对于每个样本：
              若任一时间点落在不可行区域，则 reward 为 -1，否则 reward 为 1（你也可以改成其他值）。
    """
    # 如果 drivable_map 为单张图片（形状 [H, W]），则扩展到 batch 维度
    if drivable_map.dim() == 2:
        B = traj_raster.shape[0]
        drivable_map = drivable_map.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]

    B, T, _ = traj_raster.shape
    H, W = drivable_map.shape[-2:]
    
    # 将轨迹的像素坐标四舍五入并转换为整型索引
    traj_indices = traj_raster.round().long()
    # 将索引限制在图像范围内
    traj_indices[..., 0] = traj_indices[..., 0].clamp(0, W - 1)
    traj_indices[..., 1] = traj_indices[..., 1].clamp(0, H - 1)
    
    # 构造 batch 索引
    batch_idx = torch.arange(B, device=traj_raster.device).unsqueeze(1).expand(B, T)  # [B, T]
    # 提取每个时间步对应的像素值：注意在 drivable_map 中，第一维是行 (y)，第二维是列 (x)
    pixel_values = drivable_map[batch_idx, traj_indices[..., 1], traj_indices[..., 0]]  # [B, T]
    
    # 判断每个时间步是否处于不可行区域：像素值小于 threshold 时视为 offroad
    offroad_mask = pixel_values < threshold  # [B, T]，True 表示该点不可行
    # 对每个样本，检查是否存在任一时间步 offroad
    offroad_any = offroad_mask.any(dim=1)  # [B]，True 表示该样本的轨迹中有 offroad 点

    # 如果任一时间点 offroad，则 reward 为 -1，否则为 1（你可以根据需求调整）
    reward = torch.where(offroad_any, torch.tensor(-1.0, device=traj_raster.device), torch.tensor(0.0, device=traj_raster.device))
    return reward





'''
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

'''


def transform_points_torch(points: torch.Tensor, transf_matrix: torch.Tensor) -> torch.Tensor:
    """
    使用给定的转换矩阵转换一组2D/3D点（基于 PyTorch 实现）。
    
    参数:
      points: 点数据，形状可以为 (N, F) 或 (B, N, F)，其中 F=2或3
      transf_matrix: 转换矩阵，形状为 (F+1, F+1) 或 (B, F+1, F+1)
    
    返回:
      转换后的点，形状与 points 相同，最后一维维度为 F。
      
    注意: 如果 points 是 2D 的，函数内部会扩展为 (1, N, F) 处理，然后再 squeeze。
    """
    points_log = f"received points with shape {points.shape}"
    matrix_log = f"received matrices with shape {transf_matrix.shape}"

    # 检查维度
    if points.dim() not in [2, 3]:
        raise ValueError(f"points should have 2 or 3 dimensions, {points_log}")
    if transf_matrix.dim() not in [2, 3]:
        raise ValueError(f"transf_matrix should have 2 or 3 dimensions, {matrix_log}")
    if points.dim() < transf_matrix.dim():
        raise ValueError(f"points.ndim should be >= transf_matrix.ndim, {points_log}, {matrix_log}")
    
    points_feat = points.shape[-1]
    if points_feat not in [2, 3]:
        raise ValueError(f"last dimension of points must be 2 or 3, {points_log}")
    if transf_matrix.shape[-1] != transf_matrix.shape[-2]:
        raise ValueError(f"transf_matrix must be square, {matrix_log}")
    
    matrix_feat = transf_matrix.shape[-1]
    if matrix_feat not in [3, 4]:
        raise ValueError(f"last dimension of transf_matrix must be 3 or 4, {matrix_log}")
    if points_feat != matrix_feat - 1:
        raise ValueError(f"points last dim should be one less than matrix, {points_log}, {matrix_log}")

    def _transform(points_tensor: torch.Tensor, transf_matrix_tensor: torch.Tensor) -> torch.Tensor:
        # 计算需要转换的维度数量：例如对于2D点，num_dims=2
        num_dims = transf_matrix_tensor.shape[-1] - 1
        # 将转换矩阵沿最后两个维度进行转置
        # transf_matrix_tensor: (B, F+1, F+1) -> (B, F+1, F+1)（只是调换最后两个维度）
        transf_matrix_perm = transf_matrix_tensor.permute(0, 2, 1)
        # 点转换：点乘矩阵的前 num_dims 列，再加上偏移量（矩阵最后一行）
        # points_tensor: (B, N, F), transf_matrix_perm[:, :num_dims, :num_dims]: (B, F, F)
        # transf_matrix_perm[:, -1:, :num_dims]: (B, 1, F)
        return points_tensor @ transf_matrix_perm[:, :num_dims, :num_dims] + transf_matrix_perm[:, -1:, :num_dims]

    # 情况1: points 和 transf_matrix 均为 2D
    if points.dim() == 2 and transf_matrix.dim() == 2:
        points_exp = points.unsqueeze(0)  # (1, N, F)
        matrix_exp = transf_matrix.unsqueeze(0)  # (1, F+1, F+1)
        transformed = _transform(points_exp, matrix_exp)  # (1, N, F)
        return transformed[0]
    
    # 情况2: points 和 transf_matrix 均为 3D
    elif points.dim() == 3 and transf_matrix.dim() == 3:
        return _transform(points, transf_matrix)
    
    # 情况3: points 为 3D, transf_matrix 为 2D -> 自动扩展转换矩阵
    elif points.dim() == 3 and transf_matrix.dim() == 2:
        matrix_exp = transf_matrix.unsqueeze(0)  # (1, F+1, F+1)
        return _transform(points, matrix_exp)
    
    else:
        raise NotImplementedError(f"Unsupported case: {points_log}, {matrix_log}")


