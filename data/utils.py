import numpy as np

def get_agent_lane(agent_state_np, vector_map):
    xyzh = np.array([
        agent_state_np[0],
        agent_state_np[1],
        0.0,
        agent_state_np[6]  # heading
    ])
    current_lanes = vector_map.get_current_lane(
        xyzh,
        max_dist=2.0,
        max_heading_error=np.pi/8
    )
    return current_lanes
def lane_points(element):
    vector_map = element.vec_map
    agent_from_world_tf = element.agent_from_world_tf
    agent_x = element.curr_agent_state_np[0]
    agent_y = element.curr_agent_state_np[1]
    agent_pos = np.array([agent_x, agent_y, 0.0])
    
    # 获取多条车道
    nearby_lanes = vector_map.get_lanes_within(agent_pos, dist=200.0)
    
    # 固定参数
    MAX_LANES = 20     # 最大车道数
    S_seg = 15        # 分段数
    S_point = 80       # 每段点数
    
    # 初始化固定大小的数组
    all_lane_points = np.zeros((MAX_LANES, S_seg, S_point, 3))
    
    # 处理每条车道，但不超过最大数量
    for lane_idx, lane in enumerate(nearby_lanes):
        if lane_idx >= MAX_LANES:
            break
            
        centerline = lane.center.points
        total_points = S_seg * S_point
        
        # 均匀采样点
        indices = np.linspace(0, len(centerline)-1, total_points, dtype=int)
        sampled_points = centerline[indices]
        
        # 重塑为所需形状
        lane_points = sampled_points.reshape(S_seg, S_point, -1)
        
        # 转换到智能体坐标系
        lane_points_transformed = []
        for seg in lane_points:
            homogeneous_points = np.concatenate([seg[:, :2], np.ones((len(seg), 1))], axis=1)
            transformed_points = (agent_from_world_tf @ homogeneous_points.T).T
            transformed_seg = np.concatenate([
                transformed_points[:, :2],  # x,y
                seg[:, 3:4]                # heading
            ], axis=1)
            lane_points_transformed.append(transformed_seg)
        
        lane_points_transformed = np.array(lane_points_transformed)
        all_lane_points[lane_idx] = lane_points_transformed
    
    return all_lane_points  # 返回形状 [MAX_LANES, 15, 80, 3]

def raster_vector_map(element):
    vector_map = element.vec_map
    
    # 调用rasterize函数，获取栅格图和变换矩阵
    map_img = vector_map.rasterize(
        resolution=0.5,
        return_tf_mat=False,
        incl_centerlines=True,
        incl_lane_edges=True,
        incl_lane_area=True,
        center_color=(129, 51, 255),  # 中心线颜色
        edge_color=(118, 185, 0),     # 边缘线颜色
        area_color=(214, 232, 181)    # 区域颜色
    )
    
    return map_img
def current_lanes(element):
    vector_map = element.vec_map
    curr_state = element.curr_agent_state_np
    
    xyzh = np.zeros(4)
    xyzh[0] = curr_state[0]
    xyzh[1] = curr_state[1]
    xyzh[2] = 0.0
    xyzh[3] = curr_state[6]
    
    current_lanes = vector_map.get_current_lane(
        xyzh,
        max_dist=2.0,
        max_heading_error=np.pi/8
    )
    
    lane_indices = []
    for lane in current_lanes:
        lane_indices.append(vector_map.lanes.index(lane))
    
    return np.array(lane_indices)

def get_vector_map(element):
    return element.vec_map