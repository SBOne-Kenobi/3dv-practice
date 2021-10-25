from typing import List

import numpy as np

from corners import FrameCorners
from _camtrack import PointCloudBuilder


def run_bundle_adjustment(intrinsic_mat: np.ndarray,
                          list_of_corners: List[FrameCorners],
                          max_inlier_reprojection_error: float,
                          view_mats: List[np.ndarray],
                          pc_builder: PointCloudBuilder) -> List[np.ndarray]:
    # TODO: implement
    # You may modify pc_builder using 'update_points' method
    return view_mats
