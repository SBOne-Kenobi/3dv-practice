#! /usr/bin/env python3

__all__ = [
    'CameraTrackRendererApp'
]

from collections import namedtuple
import os.path

import click
import numpy as np
from OpenGL import GLUT
from recordclass import recordclass
import transforms3d.euler

from data3d import (
    read_camera_parameters, read_poses, read_point_cloud,
    PointCloud, Pose
)
from _render import CameraTrackRenderer


def _detect_point_cloud_scale(point_cloud, percentile=70):
    point_cloud = np.asarray(point_cloud)
    center = np.average(point_cloud, axis=0)
    offsets = point_cloud - center
    offset_lens = np.linalg.norm(offsets, axis=1)
    return np.percentile(offset_lens, percentile)


def _rescale_point_cloud(point_cloud, scale):
    return np.asarray(point_cloud) * scale


def _rescale_track(cam_track, scale):
    rescaled_cam_track = []
    for pose in cam_track:
        rescaled_cam_track.append(Pose(pose.r_mat, pose.t_vec * scale))
    return rescaled_cam_track


class CameraTrackRendererApp:
    _camera_fov_y_range = namedtuple('FovRange', 'min default max')(
        np.pi / 6, np.pi / 4, np.pi / 2
    )
    _change_rates = namedtuple('ChangeRates', 'track_pos tr_xz tr_y camera')(
        80.0, 6.0, 1.0, 0.01
    )

    def __init__(self, cam_model_files,
                 tracked_cam_parameters, cam_track,
                 point_cloud):
        point_cloud_ids, point_cloud_points, point_cloud_colors = point_cloud
        if point_cloud_colors is None:
            point_cloud_colors = np.ones(point_cloud_points.shape, dtype=np.float32)
        point_cloud_scale = _detect_point_cloud_scale(point_cloud_points)
        point_cloud_target_scale = 3
        point_cloud_points = _rescale_point_cloud(
            point_cloud_points, point_cloud_target_scale / point_cloud_scale)
        cam_track = _rescale_track(
            cam_track, point_cloud_target_scale / point_cloud_scale)

        self._tracked_cam_track_len = len(cam_track)
        self._tracked_cam_track_pos_float = 0.0

        self._camera = recordclass('CameraPoseAndParameters', 'yaw pitch pos fov_y')(
            0.0, 0.0, np.array([0, 0, 10.0]), self._camera_fov_y_range.default
        )

        GLUT.glutInit()
        GLUT.glutInitWindowSize(600, 400)
        GLUT.glutInitWindowPosition(0, 0)

        self._data = recordclass('AnimationData', 'prev_time key_states last_xy')(
            GLUT.glutGet(GLUT.GLUT_ELAPSED_TIME), np.array([False] * 256), None
        )

        GLUT.glutCreateWindow(b'Camera track renderer')
        GLUT.glutDisplayFunc(self.display)
        GLUT.glutKeyboardFunc(self.key_pressed)
        GLUT.glutKeyboardUpFunc(self.key_up)
        GLUT.glutIdleFunc(self.animate)
        GLUT.glutMouseFunc(self.mouse_event)
        GLUT.glutMotionFunc(self.mouse_move)

        self._renderer_impl = CameraTrackRenderer(
            cam_model_files, tracked_cam_parameters, cam_track,
            PointCloud(point_cloud_ids, point_cloud_points, point_cloud_colors)
        )

    def animate(self):
        d_time_millis = GLUT.glutGet(GLUT.GLUT_ELAPSED_TIME) - self._data.prev_time
        self._data.prev_time += d_time_millis
        d_time = d_time_millis / 1000.0

        if self._data.key_states[ord(b'q')] or self._data.key_states[ord(b'e')]:
            if self._data.key_states[ord(b'q')]:
                self._tracked_cam_track_pos_float -= \
                    d_time * self._change_rates.track_pos
            if self._data.key_states[ord(b'e')]:
                self._tracked_cam_track_pos_float += \
                    d_time * self._change_rates.track_pos
            self._tracked_cam_track_pos_float = np.clip(
                self._tracked_cam_track_pos_float,
                0,
                self._tracked_cam_track_len - 1
            )
            self._tracked_cam_track_pos_float = np.clip(
                self._tracked_cam_track_pos_float,
                0,
                self._tracked_cam_track_len - 1
            )

        if self._data.key_states[ord(b'a')]:
            self.move_camera(d_time * np.array([-self._change_rates.tr_xz, 0, 0]))
        if self._data.key_states[ord(b'd')]:
            self.move_camera(d_time * np.array([+self._change_rates.tr_xz, 0, 0]))
        if self._data.key_states[ord(b's')]:
            self.move_camera(d_time * np.array([0, 0, +self._change_rates.tr_xz]))
        if self._data.key_states[ord(b'w')]:
            self.move_camera(d_time * np.array([0, 0, -self._change_rates.tr_xz]))

        GLUT.glutPostRedisplay()

    def key_pressed(self, key, pos_x, pos_y):
        del pos_x, pos_y
        if key == b'\033':  # esc
            GLUT.glutLeaveMainLoop()
        self._data.key_states[ord(key)] = True

    def key_up(self, key, pos_x, pos_y):
        del pos_x, pos_y
        self._data.key_states[ord(key)] = False

    def mouse_event(self, button, state, pos_x, pos_y):
        if state == GLUT.GLUT_DOWN and button in (3, 4):
            y_move_value = self._change_rates.tr_y if button == 3 else -self._change_rates.tr_y
            self.move_camera(np.array([0, y_move_value, 0]))
        if button == 1:
            if state == GLUT.GLUT_DOWN:
                self._camera.fov_y = self._camera_fov_y_range.min
            else:
                self._camera.fov_y = self._camera_fov_y_range.default
        if button == 2:
            if state == GLUT.GLUT_DOWN:
                self._camera.fov_y = self._camera_fov_y_range.max
            else:
                self._camera.fov_y = self._camera_fov_y_range.default

        if button == 0:
            if state == GLUT.GLUT_DOWN:
                self._data.last_xy = (pos_x, pos_y)
            else:
                self._data.last_xy = None

    def mouse_move(self, pos_x, pos_y):
        if self._data.last_xy is None:
            return
        change_x, change_y = \
            (pos_x - self._data.last_xy[0],
             pos_y - self._data.last_xy[1])
        self._data.last_xy = (pos_x, pos_y)
        self._camera.yaw -= change_x * self._change_rates.camera
        self._camera.yaw %= np.pi * 2
        self._camera.pitch -= change_y * self._change_rates.camera
        self._camera.pitch = np.clip(self._camera.pitch, -np.pi / 2, np.pi / 2)

    def display(self):
        self._renderer_impl.display(
            self._camera.pos,
            self.camera_rot_mat(),
            self._camera.fov_y,
            self._tracked_cam_track_pos_float
        )

    def show(self):
        self._data.prev_time = GLUT.glutGet(GLUT.GLUT_ELAPSED_TIME)
        GLUT.glutMainLoop()

    def camera_rot_mat(self):
        return transforms3d.euler.euler2mat(
            self._camera.yaw, self._camera.pitch, 0, 'ryxz'
        )

    def move_camera(self, camera_space_translation):
        world_space_translation = self.camera_rot_mat().dot(camera_space_translation)
        self._camera.pos += world_space_translation


@click.command()
@click.argument('camera_parameters', type=click.File('r'))
@click.argument('poses', type=click.File('r'))
@click.argument('point_cloud', type=click.File('r'))
def cli(camera_parameters, poses, point_cloud):
    script_path = os.path.abspath(os.path.dirname(__file__))
    camera_model_files = (os.path.join(script_path, 'camera_model/geometry.obj'),
                          os.path.join(script_path, 'camera_model/texture.jpg'))

    renderer = CameraTrackRendererApp(
        camera_model_files,
        read_camera_parameters(camera_parameters),
        read_poses(poses),
        read_point_cloud(point_cloud)
    )
    renderer.show()


if __name__ == '__main__':
    cli()  # pylint:disable=no-value-for-parameter
