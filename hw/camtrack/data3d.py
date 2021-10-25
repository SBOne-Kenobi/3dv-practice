#! /usr/bin/env python3

__all__ = [
    'DataFormatError',
    'CameraParameters',
    'read_camera_parameters',
    'write_camera_parameters',
    'Pose',
    'read_poses',
    'write_poses',
    'PointCloud',
    'read_point_cloud',
    'write_point_cloud'
]

from collections import namedtuple
import copy
import itertools as itt
from typing import IO, List
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


import click
from good import All, Invalid, Schema, Range, Required
import numpy as np


class DataFormatError(Exception):

    pass


def _check_data_format(stream, schema, name):
    try:
        data = yaml.load(stream, Loader)
    except yaml.YAMLError as err:
        raise DataFormatError('{} YAML error: {}'.format(name, err))
    try:
        return schema(data)
    except Invalid as err:
        raise DataFormatError('{} format error: {}'.format(name, err))


def _check_and_write_data(data, stream, schema, name):
    try:
        schema(copy.deepcopy(data))
    except Invalid as err:
        raise DataFormatError('{} format error: {}'.format(name, err))
    yaml.dump(data, stream, Dumper)


CameraParameters = namedtuple('CameraParameters', ('fov_y', 'aspect_ratio'))


CAMERA_PARAMETERS_SCHEMA = Schema({
    Required('camera'): {
        Required('fov_y'): All(float, Range(min=0)),
        Required('aspect_ratio'): All(float, Range(min=0))
    }
})


def read_camera_parameters(stream: IO[str]) -> CameraParameters:
    data = _check_data_format(stream, CAMERA_PARAMETERS_SCHEMA, 'Camera')
    return CameraParameters(data['camera']['fov_y'],
                            data['camera']['aspect_ratio'])


def write_camera_parameters(camera_parameters: CameraParameters,
                            stream: IO[str]) -> None:
    data = {
        'camera': {
            'fov_y': camera_parameters.fov_y,
            'aspect_ratio': camera_parameters.aspect_ratio
        }
    }
    _check_and_write_data(data, stream, CAMERA_PARAMETERS_SCHEMA, 'Camera')


def _all_close(mat_1, mat_2):
    return np.allclose(mat_1, mat_2, atol=1.e-5, rtol=0)


def _to_orthogonal(mat3x3):
    u_mat, _, vh_mat = np.linalg.svd(mat3x3)
    return u_mat @ vh_mat


def _check_rotation_mat(mat):
    mat = np.array(mat, dtype=np.float64)
    if mat.shape != (3, 3) or not _all_close(np.eye(3), mat.dot(mat.T)):
        raise Invalid('Invalid rotation matrix')
    return _to_orthogonal(mat)


def _check_3d_vec(vec):
    vec = np.array(vec, dtype=np.float64)
    if vec.shape != (3,):
        raise Invalid('Invalid 3D vector')
    return vec


def _check_rgb_vec(vec):
    vec = _check_3d_vec(vec)
    if not np.all(np.logical_and(vec >= 0, vec <= 1)):
        raise Invalid('Color vector values must lie within [0, 1]')
    return vec


POSE_SCHEMA = Schema({
    Required('R'): _check_rotation_mat,
    Required('t'): _check_3d_vec
})


POSES_SCHEMA = Schema({
    Required('frames'): [{
        Required('frame'): All(int, Range(min=0)),
        Required('pose'): POSE_SCHEMA
    }]
})


Pose = namedtuple('Pose', ('r_mat', 't_vec'))


def read_poses(stream: IO[str]) -> List[Pose]:
    dict_poses = _check_data_format(stream, POSES_SCHEMA, 'Poses')['frames']
    dict_poses.sort(key=lambda p: p['frame'])
    return [Pose(p['pose']['R'], p['pose']['t']) for p in dict_poses]


def _convert_pose_to_dict(pose):
    return {
        'R': pose.r_mat.tolist(),
        't': pose.t_vec.tolist()
    }


def write_poses(poses: List[Pose], stream: IO[str]) -> None:
    data = {'frames': [{'frame': frame, 'pose': _convert_pose_to_dict(pose)}
                       for frame, pose in enumerate(poses, 1)]}
    _check_and_write_data(data, stream, POSES_SCHEMA, 'Poses')


PointCloud = namedtuple('PointCloud', ('ids', 'points', 'colors'))


POINT_CLOUD_SCHEMA = Schema({
    Required('points'): [{
        Required('id'): All(int, Range(min=0)),
        Required('point'): _check_3d_vec,
        'color': _check_rgb_vec
    }]
})


def read_point_cloud(stream: IO[str]) -> PointCloud:
    data = _check_data_format(stream, POINT_CLOUD_SCHEMA, 'Point cloud')
    data_list = data['points']
    data_list.sort(key=lambda item: item['id'])

    ids = np.array([item['id'] for item in data_list]).reshape((-1, 1))
    points = np.array([item['point'] for item in data_list])

    colors = [item.get('color') for item in data_list]
    colors_mask = [c is None for c in colors]
    if any(colors_mask):
        if all(colors_mask):
            colors = None
        else:
            raise DataFormatError(
                "Colors can't be defined not for all points of point cloud"
            )
    else:
        colors = np.array(colors)

    return PointCloud(ids, points, colors)


def write_point_cloud(point_cloud: PointCloud, stream: IO[str]) -> None:
    ids = map(int, point_cloud.ids.flatten())
    points = (p.tolist() for p in point_cloud.points)
    if point_cloud.colors is None:
        colors = itt.repeat(None)
    else:
        colors = (c.tolist() for c in point_cloud.colors)
    keys = ('id', 'point', 'color')
    data = {
        'points': [{k: v for k, v in zip(keys, values) if v is not None}
                   for values in zip(ids, points, colors)]
    }
    _check_and_write_data(data, stream, POINT_CLOUD_SCHEMA, 'Point cloud')


@click.group()
def _cli():
    pass


@_cli.command('poses')
@click.argument('yaml_file', type=click.File('r'))
def _check_poses(yaml_file):
    poses = read_poses(yaml_file)
    click.echo('Correct poses file format: {} frames'.format(len(poses)))


@_cli.command('camera')
@click.argument('yaml_file', type=click.File('r'))
def _check_camera(yaml_file):
    camera = read_camera_parameters(yaml_file)
    click.echo('Correct camera file format: '
               'fov_y = {}, aspect_ratio = {}'.format(*camera))


@_cli.command('cloud')
@click.argument('yaml_file', type=click.File('r'))
def _check_point_cloud(yaml_file):
    point_cloud = read_point_cloud(yaml_file)
    click.echo('Correct point cloud file format: {} points {} colors'.format(
        len(point_cloud.points),
        'without' if point_cloud.colors is None else 'with'
    ))


if __name__ == '__main__':
    _cli()
