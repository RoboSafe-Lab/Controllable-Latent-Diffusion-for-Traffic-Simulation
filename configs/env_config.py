
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from configs.rasterizer_config import RasterizerConfig

@dataclass
class NuscTrajdataEnvConfig:
    data_generation_trajdata_centric: str = "agent"  # "agent", "scene"
    data_generation_trajdata_only_types: List[str] = field(default_factory=lambda: ["vehicle"])
    data_generation_trajdata_predict_types: List[str] = field(default_factory=lambda: ["vehicle"])
    data_generation_trajdata_scene_desc_contains: Optional[str] = None
    data_generation_trajdata_incl_map: bool = True
    data_generation_trajdata_max_agents_distance: float = np.inf
    data_generation_trajdata_standardize_data: bool = True
    data_generation_trajdata_other_agents_num: Optional[int] = None

    # 添加 rasterizer 子配置
    rasterizer: RasterizerConfig = RasterizerConfig()