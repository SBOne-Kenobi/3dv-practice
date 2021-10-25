#! /usr/bin/env python3

__all__ = [
    'start_from_origin',
    'get_all_translations',
    'get_all_rotation_mats',
    'calc_track_length',
    'calc_translation_errors',
    'calc_rotation_error_rad',
    'calc_rotation_errors_rad',
    'calc_errors',
    'calc_auc',
    'calc_vol_under_surface',
    'MAX_ROTATION_ERR_RAD',
    'MAX_TRANSLATION_ERR'
]

from typing import List, Tuple

import click
import numpy as np
from transforms3d.axangles import mat2axangle

from data3d import Pose, read_poses


def _to_mat4x4(pose):
    return np.vstack((np.hstack((pose.r_mat, pose.t_vec.reshape(-1, 1))),
                      np.array([0, 0, 0, 1.0])))


def _to_pose_from_mat4x4(mat):
    return Pose(mat[:3, :3], mat[:3, 3].flatten())


def start_from_origin(poses: List[Pose]) -> List[Pose]:
    mat_0_inv = np.linalg.inv(_to_mat4x4(poses[0]))
    return [_to_pose_from_mat4x4(mat_0_inv @ _to_mat4x4(p)) for p in poses]


def get_all_translations(poses: List[Pose]) -> np.ndarray:
    return np.array([p.t_vec for p in poses])


def get_all_rotation_mats(poses: List[Pose]) -> np.ndarray:
    return np.array([p.r_mat for p in poses])


def calc_track_length(t_vecs: np.ndarray) -> float:
    diffs = t_vecs[1:, :] - t_vecs[:-1, :]
    return np.linalg.norm(diffs, axis=1).sum()


def calc_translation_errors(ground_truth_t_vecs: np.ndarray,
                            estimate_t_vecs: np.ndarray) -> np.ndarray:
    scale, _, _, _ = np.linalg.lstsq(
        estimate_t_vecs.reshape((-1, 1)),
        ground_truth_t_vecs.flatten(),
        rcond=None
    )
    scale = np.abs(scale.item())
    scaled_estimate_t_vecs = scale * estimate_t_vecs
    ground_truth_track_length = calc_track_length(ground_truth_t_vecs)
    return np.linalg.norm(ground_truth_t_vecs - scaled_estimate_t_vecs,
                          axis=1) / ground_truth_track_length


def calc_rotation_error_rad(r_mat_1: np.ndarray, r_mat_2: np.ndarray) -> float:
    r_mat_diff = r_mat_2 @ r_mat_1.T
    _, angle = mat2axangle(r_mat_diff)
    return np.abs(angle)


def calc_rotation_errors_rad(r_mats_1: np.ndarray,
                             r_mats_2: np.ndarray) -> np.ndarray:
    return np.array([calc_rotation_error_rad(r_mat_1, r_mat_2)
                     for r_mat_1, r_mat_2 in zip(r_mats_1, r_mats_2)])


def calc_errors(ground_truth_track: List[Pose],
                estimate_track: List[Pose]) -> Tuple[np.ndarray, np.ndarray]:
    ground_truth_track = start_from_origin(ground_truth_track)
    estimate_track = start_from_origin(estimate_track)
    r_errors = calc_rotation_errors_rad(
        get_all_rotation_mats(ground_truth_track),
        get_all_rotation_mats(estimate_track),
    )
    t_errors = calc_translation_errors(
        get_all_translations(ground_truth_track),
        get_all_translations(estimate_track),
    )
    return r_errors, t_errors


def calc_auc(errors: np.ndarray, max_error: float) -> float:
    return ((1.0 - np.minimum(1.0, errors / max_error)) / errors.size).sum()


def _build_error_curve(errors: np.ndarray, max_error: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    cost = 1.0 / errors.size
    sorted_errors = np.sort(errors)
    size = np.searchsorted(sorted_errors, max_error)
    # pylint:disable=invalid-name
    x = sorted_errors[:size]
    y = np.cumsum(np.full_like(x, cost))
    x = np.concatenate((np.zeros((1,)), x, np.full((1,), max_error)))
    y = np.concatenate((np.zeros((1,)), y, np.full((1,), y[-1])))
    return x, y


MAX_ROTATION_ERR_RAD = np.pi / 8
MAX_ROTATION_ERR_DEG = np.rad2deg(MAX_ROTATION_ERR_RAD)
MAX_TRANSLATION_ERR = 0.25


def calc_vol_under_surface(r_errors: np.ndarray, t_errors: np.ndarray,
                           max_r_error: float = MAX_ROTATION_ERR_RAD,
                           max_t_error: float = MAX_TRANSLATION_ERR) -> float:
    tmp_1 = 1.0 - np.minimum(1.0, r_errors / max_r_error)
    tmp_2 = 1.0 - np.minimum(1.0, t_errors / max_t_error)
    return (tmp_1 * tmp_2).sum() / r_errors.size


@click.command()
@click.argument('ground_truth_file', type=click.File('r'))
@click.argument('estimate_file', type=click.File('r'))
@click.option('--plot', '-p', is_flag=True, help='Plot frame errors')
def _cli(ground_truth_file, estimate_file, plot):
    # pylint:disable=too-many-locals

    gt_track = read_poses(ground_truth_file)
    e_track = read_poses(estimate_file)
    r_errors, t_errors = calc_errors(gt_track, e_track)
    r_errors = np.degrees(r_errors)  # pylint:disable=assignment-from-no-return

    r_max = r_errors.max()
    r_median = np.median(r_errors)
    r_auc = calc_auc(r_errors, MAX_ROTATION_ERR_DEG)
    t_max = t_errors.max()
    t_median = np.median(t_errors)
    t_auc = calc_auc(t_errors, MAX_TRANSLATION_ERR)

    r_line_template = 'Rotation errors (degrees)' \
                      '\n  max = {}\n  median = {}\n  AUC = {}'
    click.echo(r_line_template.format(r_max, r_median, r_auc))
    t_line_template = 'Translation errors' \
                      '\n  max = {}\n  median = {}\n  AUC = {}'
    click.echo(t_line_template.format(t_max, t_median, t_auc))
    click.echo('Volume under surface = {}'.format(
        calc_vol_under_surface(r_errors, t_errors, MAX_ROTATION_ERR_DEG)
    ))

    if not plot:
        return

    import matplotlib.pyplot as plt  # pylint:disable=import-outside-toplevel
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(r_errors, 'g')
    plt.xlim([0, r_errors.size])
    plt.xlabel('Frame')
    plt.ylabel('Rotation error (degrees)')
    plt.subplot(2, 2, 2)
    plt.plot(*_build_error_curve(r_errors, MAX_ROTATION_ERR_DEG), 'r')
    plt.xlim([0.0, MAX_ROTATION_ERR_DEG])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Rotation error (degrees)')
    plt.ylabel('Frame rate')
    plt.subplot(2, 2, 3)
    plt.plot(t_errors, 'g')
    plt.xlim([0, t_errors.size])
    plt.xlabel('Frame')
    plt.ylabel('Translation error')
    plt.subplot(2, 2, 4)
    plt.plot(*_build_error_curve(t_errors, MAX_TRANSLATION_ERR), 'r')
    plt.xlim([0.0, MAX_TRANSLATION_ERR])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Translation error')
    plt.ylabel('Frame rate')
    plt.show()


if __name__ == '__main__':
    _cli()  # pylint:disable=no-value-for-parameter
