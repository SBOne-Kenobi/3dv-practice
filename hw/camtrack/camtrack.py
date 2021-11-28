#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp
from cv2 import cv2

import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4,
    check_inliers_mask,
    check_baseline,
    Correspondences,
    eye3x4,
    triangulate_ransac
)
from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose

PnPParameters = namedtuple(
    'PnPParameters',
    ('inliers_prob', 'outliers_ratio', 'max_reprojection_error',
     'min_inlier_count', 'min_inlier_ratio')
)


def _find_view_mat_on_inliers(points3d, points2d, intrinsic_mat, r_vec_init, t_vec_init):
    success, r_vec, t_vec = cv2.solvePnP(
        points3d, points2d,
        intrinsic_mat, np.array([]),
        r_vec_init, t_vec_init, True,
        cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        r_vec = r_vec_init
        t_vec = t_vec_init
    return rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)


def _find_view_mat_pnp(cloud: PointCloudBuilder,
                       corners: FrameCorners,
                       intrinsic_mat: np.ndarray,
                       params: PnPParameters) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    ids, (ind1, ind2) = snp.intersect(cloud.ids.flatten(), corners.ids.flatten(), indices=True)

    if len(ids) < 4:
        return None, None, None

    iterations = int(np.ceil(np.log(1.0 - params.inliers_prob) / np.log(1.0 - (1.0 - params.outliers_ratio) ** 4)))

    success, r_vec, t_vec, inliers = cv2.solvePnPRansac(
        cloud.points[ind1],
        corners.points[ind2],
        intrinsic_mat,
        np.array([]),
        iterationsCount=iterations,
        reprojectionError=params.max_reprojection_error,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not success:
        return None, None, None
    inliers = inliers.flatten()
    inliers_mask = np.full_like(ids, False)
    inliers_mask[inliers] = True
    if not check_inliers_mask(inliers_mask, params.min_inlier_count, params.min_inlier_ratio):
        return None, None, None

    return _find_view_mat_on_inliers(
        cloud.points[ind1[inliers]],
        corners.points[ind2[inliers]],
        intrinsic_mat, r_vec, t_vec
    ), ids[~inliers_mask], len(inliers)


InitViewsParameters = namedtuple(
    'InitViewsParameters',
    ('inliers_prob', 'max_distance_to_epipolar_line',
     'outliers_ratio', 'triangulate_params',
     'max_frame_distance', 'max_reprojection_error',
     'min_inlier_count', 'min_inlier_ratio')
)


def find_view_mat(correspondences: Correspondences,
                  intrinsic_mat: np.ndarray,
                  params: InitViewsParameters) \
        -> Tuple[Optional[np.ndarray], Optional[float], Optional[int], Optional[float]]:
    if len(correspondences.ids) < 5:
        return None, None, None, None

    iterations_e_mat = int(np.ceil(np.log(1.0 - params.inliers_prob) / np.log(1.0 - (1.0 - params.outliers_ratio) ** 5)))

    e_mat, inliers = cv2.findEssentialMat(
        correspondences.points_1,
        correspondences.points_2,
        intrinsic_mat,
        cv2.RANSAC,
        params.inliers_prob,
        params.max_distance_to_epipolar_line,
        iterations_e_mat
    )

    iterations_hom = int(np.ceil(np.log(1.0 - params.inliers_prob) / np.log(params.outliers_ratio)))

    _, inliers_hom = cv2.findHomography(
        correspondences.points_1,
        correspondences.points_2,
        cv2.RANSAC,
        params.max_reprojection_error,
        maxIters=iterations_hom
    )

    inliers = inliers.flatten()
    inliers_hom = inliers_hom.flatten()

    if not check_inliers_mask(inliers, params.min_inlier_count, params.min_inlier_ratio):
        return None, None, None, None

    correspondences = Correspondences(
        correspondences.ids[inliers],
        correspondences.points_1[inliers],
        correspondences.points_2[inliers]
    )

    r_1, r_2, _t = cv2.decomposeEssentialMat(e_mat)
    best_view_mat = None
    best_point_count = None
    best_med_cos = None
    for R in [r_1, r_2]:
        for t in [-_t, _t]:
            view_mat = np.hstack([R, t.reshape(-1, 1)])
            points3d, corr_ids, med_cos = triangulate_correspondences(
                correspondences, eye3x4(), view_mat,
                intrinsic_mat, params.triangulate_params
            )

            point_count = len(corr_ids)

            if best_view_mat is None or point_count > best_point_count:
                best_view_mat = view_mat
                best_point_count = point_count
                best_med_cos = med_cos

    return best_view_mat, best_med_cos, best_point_count, np.sum(inliers_hom) / np.sum(inliers)


def _init_views_impl(intrinsic_mat, corner_storage, params, min_point_count, max_hom_ratio):
    for ln in range(params.max_frame_distance, 0, -1):
        for i in range(len(corner_storage) - ln):
            j = i + ln
            view_mat, med_cos, point_count, hom_ratio = find_view_mat(
                build_correspondences(
                    corner_storage[i],
                    corner_storage[j]
                ), intrinsic_mat, params)
            if view_mat is None:
                continue
            if point_count < min_point_count:
                continue
            if hom_ratio > max_hom_ratio:
                continue
            return (
                (i, view_mat3x4_to_pose(eye3x4())),
                (j, view_mat3x4_to_pose(view_mat))
            )
    return None


def init_views(intrinsic_mat: np.ndarray,
               corner_storage: CornerStorage,
               params: InitViewsParameters) \
        -> Tuple[Tuple[int, Pose], Tuple[int, Pose]]:
    max_hom_ratio = 0.2
    while max_hom_ratio <= 1.0:
        min_point_count = 200
        while min_point_count > 20:
            res = _init_views_impl(
                intrinsic_mat, corner_storage,
                params, min_point_count, max_hom_ratio
            )
            if res is not None:
                return res
            min_point_count //= 2
        max_hom_ratio *= 2.0
    raise RuntimeError("Not found initial views")


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    view_mats: list[Optional[np.ndarray]] = [None] * frame_count
    point_cloud_builder = PointCloudBuilder()

    max_reprojection_error = 5.0

    triangulate_params = TriangulationParameters(
        max_reprojection_error=max_reprojection_error,
        min_triangulation_angle_deg=1.0,
        min_depth=0.1,
    )
    pnp_params = PnPParameters(
        inliers_prob=0.999,
        outliers_ratio=0.5,
        max_reprojection_error=max_reprojection_error,
        min_inlier_count=5,
        min_inlier_ratio=0.3,
    )
    base_line_min_dist = 0
    max_frame_distance = 70
    init_views_params = InitViewsParameters(
        inliers_prob=0.999,
        outliers_ratio=0.5,
        max_distance_to_epipolar_line=1,
        triangulate_params=triangulate_params,
        max_frame_distance=70,
        max_reprojection_error=max_reprojection_error,
        min_inlier_count=5,
        min_inlier_ratio=0.3,
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = init_views(
            intrinsic_mat, corner_storage,
            init_views_params
        )

    calculated_frames = 2

    frame1 = known_view_1[0]
    frame2 = known_view_2[0]
    view_mat1 = pose_to_view_mat3x4(known_view_1[1])
    view_mat2 = pose_to_view_mat3x4(known_view_2[1])
    view_mats[frame1] = view_mat1
    view_mats[frame2] = view_mat2

    all_view_mats = [(frame2, view_mats[frame2])]

    outlier_ids = None

    count_range = set()

    def get_range(index):
        return range(
            max(0, index - max_frame_distance),
            min(frame_count, index + max_frame_distance + 1)
        )

    def update_range(index):
        for ii in get_range(index):
            count_range.add(ii)

    update_range(frame1)
    update_range(frame2)
    print('Initialize:')
    print(f'\tFrames: {frame1}, {frame2}')

    while calculated_frames < frame_count:
        # triangulation
        # if frame2 is None:
        #     best_points3d, best_corr_ids, best_med_cos = None, None, None
        #     for i in get_range(frame1):
        #         if i == frame1 or view_mats[i] is None:
        #             continue
        #         if not check_baseline(view_mats[frame1], view_mats[i], base_line_min_dist):
        #             continue
        #         corr = build_correspondences(corner_storage[frame1], corner_storage[i], outlier_ids)
        #         points3d, corr_ids, med_cos = triangulate_correspondences(
        #             corr, view_mats[frame1], view_mats[i],
        #             intrinsic_mat, triangulate_params
        #         )
        #         if frame2 is None or best_med_cos > med_cos:
        #             frame2 = i
        #             best_points3d = points3d
        #             best_corr_ids = corr_ids
        #             best_med_cos = med_cos
        # else:
        #     corr = build_correspondences(corner_storage[frame1], corner_storage[frame2], outlier_ids)
        #     best_points3d, best_corr_ids, best_med_cos = triangulate_correspondences(
        #         corr, view_mats[frame1], view_mats[frame2],
        #         intrinsic_mat, triangulate_params
        #     )
        best_points3d, best_corr_ids, best_med_cos = triangulate_ransac(
            corner_storage, (frame1, view_mats[frame1]), all_view_mats,
            intrinsic_mat, triangulate_params,
            0.999, 0.5, 5, 0.3, outlier_ids
        )
        all_view_mats.append((frame1, view_mats[frame1]))
        point_cloud_builder.add_points(best_corr_ids, best_points3d)
        print(f'\tTriangulated {len(best_points3d)} points')
        print(f'\tCloud size: {len(point_cloud_builder.ids)}')

        # calculate motion
        best_frame = -1
        best_frame_view_mat = None
        best_frame_outlier_ids = None
        best_frame_inliers_count = 0
        for _ in range(2):
            for i in count_range:
                if view_mats[i] is not None:
                    continue
                view_mat, outlier_ids, inliers_count = _find_view_mat_pnp(
                    point_cloud_builder, corner_storage[i],
                    intrinsic_mat, pnp_params
                )
                if view_mat is not None:
                    if best_frame == -1 or best_frame_inliers_count < inliers_count:
                        best_frame = i
                        best_frame_inliers_count = inliers_count
                        best_frame_view_mat = view_mat
                        best_frame_outlier_ids = outlier_ids
            if best_frame != -1:
                break
            if pnp_params.min_inlier_ratio == 0.0:
                break
            pnp_params = PnPParameters(
                inliers_prob=pnp_params.inliers_prob,
                outliers_ratio=pnp_params.outliers_ratio,
                max_reprojection_error=pnp_params.max_reprojection_error,
                min_inlier_count=0,
                min_inlier_ratio=0.0,
            )
        if best_frame == -1:
            break
        view_mats[best_frame] = best_frame_view_mat
        outlier_ids = best_frame_outlier_ids
        update_range(best_frame)
        frame1 = best_frame
        frame2 = None
        calculated_frames += 1
        print(f'Frame {frame1}:')
        print(f'\tInliers: {best_frame_inliers_count}')

    none_views = [i for i, val in enumerate(view_mats) if val is None]
    if len(none_views) > 0:
        raise RuntimeError(f"{len(none_views)} views can't be calculated.")

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        max_reprojection_error
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
