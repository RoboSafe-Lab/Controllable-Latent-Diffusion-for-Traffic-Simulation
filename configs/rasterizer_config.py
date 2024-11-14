from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class RasterizerConfig:
    num_sem_layers: int = 3
    rgb_idx_groups: List[List[int]] = field(default_factory=lambda: [[0], [1], [2]])
    raster_size: int = 224
    pixel_size: float = 0.5
    ego_center: Tuple[float, float] = (-0.5, 0.0)
    no_map_fill_value: float = -1.0