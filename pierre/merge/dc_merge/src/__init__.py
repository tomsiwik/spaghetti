"""DC-Merge: Directional-Consistent Model Merging (MLX implementation).

Paper: arXiv 2603.06242
"""

from pierre.merge.dc_merge.src.merge import dc_merge, dc_merge_from_config
from pierre.merge.dc_merge.src.model import (
    energy_smoothing,
    construct_cover_basis,
    project_to_cover_space,
    project_to_param_space,
)
