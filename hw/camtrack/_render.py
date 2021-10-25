#! /usr/bin/env python3

__all__ = [
    'CameraTrackRenderer'
]

from collections import namedtuple
from contextlib import ExitStack
from typing import List, Tuple

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL import GLUT
from OpenGL.arrays import vbo
from PIL import Image
from pyquaternion import Quaternion
import transforms3d

import data3d


def _build_point_cloud_program():
    vertex_shader = shaders.compileShader(
        """
        #version 120
        uniform mat4 mvp;
        attribute vec3 in_position;
        attribute vec3 in_color;
        varying vec3 color;
        void main() {
            color = in_color;
            gl_Position = mvp * vec4(in_position, 1.0);
            gl_PointSize = 2.0;
        }""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 120
        
        varying vec3 color;
        
        void main() {
            gl_FragColor = vec4(color, 1.0);
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(vertex_shader, fragment_shader)


def _build_cam_track_line_program():
    vertex_shader = shaders.compileShader(
        """
        #version 120
        uniform mat4 mvp;
        attribute vec3 in_position;
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
        }""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 120
        
        void main() {
            gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(vertex_shader, fragment_shader)


def _build_cam_model_program():
    vertex_shader = shaders.compileShader(
        """
        #version 120
        uniform mat4 mvp;
        attribute vec2 in_uv;
        attribute vec3 in_position;
        
        varying vec2 uv;
        
        void main() {
            uv = in_uv;
            gl_Position = mvp * vec4(in_position, 1.0);
        }""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 120
        uniform sampler2D texture_sampler;
        
        varying vec2 uv;
        
        void main() {
            gl_FragColor = vec4(texture2D(texture_sampler, uv).rgb, 1.0);
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(vertex_shader, fragment_shader)


def _build_cam_frustrum_program():
    vertex_shader = shaders.compileShader(
        """
        #version 120
        uniform mat4 mvp;
        attribute vec3 in_position;
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
        }""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 120
        
        void main() {
            gl_FragColor = vec4(1.0, 1.0, 0.0, 1.0);
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(vertex_shader, fragment_shader)


ObjModel = namedtuple('ObjModel', ('vertices', 'normals', 'uvs', 'faces'))


def _load_obj(path: str) -> ObjModel:
    with open(path, "rt") as stream:
        lines = stream.readlines()

    vertices = np.array([line.split()[1:] for line in lines if line.startswith("v ")], dtype=np.float32)
    normals = np.array([line.split()[1:] for line in lines if line.startswith("vn ")], dtype=np.float32)
    uvs = np.array([line.split()[1:] for line in lines if line.startswith("vt ")], dtype=np.float32)

    def _parse_face(line):
        face_vertex_ids = [id - 1 if id > 0 else id + len(vertices) for id in map(int, line.split()[1:])]
        return face_vertex_ids

    faces = np.array([_parse_face(line) for line in lines if line.startswith("f ")], dtype=np.int32)

    return ObjModel(vertices, normals, uvs, faces)


def _load_jpg_texture(path: str) -> np.ndarray:
    image = Image.open(path)
    rgb_array = np.array(image.getdata(), dtype=np.uint8).reshape((image.size[0], image.size[1], 3))
    return np.flip(rgb_array, axis=0)


def _create_gl_texture(pixels: np.ndarray) -> GL.GLuint:
    with ExitStack() as stack:
        id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, id)
        stack.callback(lambda: GL.glBindTexture(GL.GL_TEXTURE_2D, 0))

        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0,  # target, level of detail
                        GL.GL_RGB8,  # internal format
                        pixels.shape[1], pixels.shape[0], 0,  # width, height, border
                        GL.GL_RGB, GL.GL_UNSIGNED_BYTE,  # external format, type
                        pixels)

        return id


def _extend_rotation_matr(mat3: np.ndarray) -> np.ndarray:
    return np.matrix([
        [mat3[0][0], mat3[0][1], mat3[0][2], 0],
        [mat3[1][0], mat3[1][1], mat3[1][2], 0],
        [mat3[2][0], mat3[2][1], mat3[2][2], 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)


def _get_pose_matrix(pose: data3d.Pose) -> np.ndarray:
    translation = np.matrix([
        [1, 0, 0, pose.t_vec[0]],
        [0, 1, 0, pose.t_vec[1]],
        [0, 0, 1, pose.t_vec[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    rotation = _extend_rotation_matr(pose.r_mat)

    return translation.dot(rotation)


_opencv_rotation_matrix = transforms3d.euler.euler2mat(np.pi, 0, 0, "rxyz")


def _from_opencv_format(points):
    return np.array([_opencv_rotation_matrix.dot(point) for point in points], dtype=np.float32)


class _PointCloudRenderer:

    def __init__(self, point_cloud: data3d.PointCloud):
        self._point_positions = vbo.VBO(_from_opencv_format(np.array(point_cloud.points, dtype=np.float32)))
        self._point_colors = vbo.VBO(np.array(point_cloud.colors, dtype=np.float32))
        self._point_cloud_program = _build_point_cloud_program()

    def render(self, mvp: np.ndarray):
        with ExitStack() as stack:
            shaders.glUseProgram(self._point_cloud_program)
            stack.callback(lambda: shaders.glUseProgram(0))

            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self._point_cloud_program, 'mvp'),
                1, True, mvp)

            self._point_positions.bind()
            position_loc = GL.glGetAttribLocation(self._point_cloud_program, 'in_position')
            GL.glEnableVertexAttribArray(position_loc)
            stack.callback(lambda: GL.glDisableVertexAttribArray(position_loc))
            GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                     False, 0,
                                     self._point_positions)
            self._point_positions.unbind()

            self._point_colors.bind()
            color_loc = GL.glGetAttribLocation(self._point_cloud_program, 'in_color')
            GL.glEnableVertexAttribArray(color_loc)
            stack.callback(lambda: GL.glDisableVertexAttribArray(color_loc))
            GL.glVertexAttribPointer(color_loc, 3, GL.GL_FLOAT,
                                     False, 0,
                                     self._point_colors)
            self._point_colors.unbind()

            GL.glDrawArrays(GL.GL_POINTS, 0, len(self._point_positions))


class _CameraTrackLineRenderer:

    def __init__(self, tracked_cam_track: List[data3d.Pose]):
        self._cam_track_line_positions = vbo.VBO(
            _from_opencv_format(np.array([pose.t_vec for pose in tracked_cam_track], dtype=np.float32)))
        self._cam_track_line_program = _build_cam_track_line_program()

    def render(self, mvp):
        with ExitStack() as stack:
            shaders.glUseProgram(self._cam_track_line_program)
            stack.callback(lambda: shaders.glUseProgram(0))

            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self._cam_track_line_program, 'mvp'),
                1, True, mvp)

            self._cam_track_line_positions.bind()
            position_loc = GL.glGetAttribLocation(self._cam_track_line_program, 'in_position')
            GL.glEnableVertexAttribArray(position_loc)
            stack.callback(lambda: GL.glDisableVertexAttribArray(position_loc))
            GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                     False, 0,
                                     self._cam_track_line_positions)
            self._cam_track_line_positions.unbind()

            GL.glDrawArrays(GL.GL_LINE_STRIP, 0, len(self._cam_track_line_positions))


class _CameraModelRenderer:

    def __init__(self, cam_model_files: Tuple[str, str]):

        cam_model = _load_obj(cam_model_files[0])

        self._cam_model_vertex_positions = vbo.VBO(np.array([
            [cam_model.vertices[v_id] for v_id in face]
            for face in cam_model.faces
        ], dtype=np.float32).reshape(-1, 3))
        self._cam_model_uvs = vbo.VBO(np.array([
            [cam_model.uvs[v_id] for v_id in face]
            for face in cam_model.faces
        ], dtype=np.float32).reshape(-1, 2))

        self._cam_model_texture = _create_gl_texture(_load_jpg_texture(cam_model_files[1]))

        self._cam_model_program = _build_cam_model_program()

    def render(self, mvp):
        with ExitStack() as stack:
            shaders.glUseProgram(self._cam_model_program)
            stack.callback(lambda: shaders.glUseProgram(0))

            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self._cam_model_program, 'mvp'),
                1, True, mvp)

            self._cam_model_vertex_positions.bind()
            position_loc = GL.glGetAttribLocation(self._cam_model_program, 'in_position')
            GL.glEnableVertexAttribArray(position_loc)
            stack.callback(lambda: GL.glDisableVertexAttribArray(position_loc))
            GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                     False, 0,
                                     self._cam_model_vertex_positions)
            self._cam_model_vertex_positions.unbind()

            self._cam_model_uvs.bind()
            uv_loc = GL.glGetAttribLocation(self._cam_model_program, 'in_uv')
            GL.glEnableVertexAttribArray(uv_loc)
            stack.callback(lambda: GL.glDisableVertexAttribArray(uv_loc))
            GL.glVertexAttribPointer(uv_loc, 2, GL.GL_FLOAT,
                                     False, 0,
                                     self._cam_model_uvs)
            self._cam_model_uvs.unbind()

            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._cam_model_texture)
            GL.glUniform1i(GL.glGetUniformLocation(self._cam_model_program, 'texture_sampler'), 0)

            GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(self._cam_model_vertex_positions))


class _CameraFrustrumRenderer:

    def __init__(self, tracked_cam_parameters: data3d.CameraParameters):
        near = 1
        far = 22
        y_tan = np.tan(tracked_cam_parameters.fov_y / 2)
        x_tan = tracked_cam_parameters.aspect_ratio * y_tan
        self._cam_frustrum_positions = vbo.VBO(np.array([
            # near square
            [[x_tan * near, y_tan * near, -near], [x_tan * near, -y_tan * near, -near]],
            [[x_tan * near, -y_tan * near, -near], [-x_tan * near, -y_tan * near, -near]],
            [[-x_tan * near, -y_tan * near, -near], [-x_tan * near, y_tan * near, -near]],
            [[-x_tan * near, y_tan * near, -near], [x_tan * near, y_tan * near, -near]],
            # far square
            [[x_tan * far, y_tan * far, -far], [x_tan * far, -y_tan * far, -far]],
            [[x_tan * far, -y_tan * far, -far], [-x_tan * far, -y_tan * far, -far]],
            [[-x_tan * far, -y_tan * far, -far], [-x_tan * far, y_tan * far, -far]],
            [[-x_tan * far, y_tan * far, -far], [x_tan * far, y_tan * far, -far]],
            # connecting lines
            [[x_tan * near, y_tan * near, -near], [x_tan * far, y_tan * far, -far]],
            [[x_tan * near, -y_tan * near, -near], [x_tan * far, -y_tan * far, -far]],
            [[-x_tan * near, -y_tan * near, -near], [-x_tan * far, -y_tan * far, -far]],
            [[-x_tan * near, y_tan * near, -near], [-x_tan * far, y_tan * far, -far]],
        ], dtype=np.float32).flatten())
        self._cam_frustrum_program = _build_cam_frustrum_program()

    def render(self, mvp):
        with ExitStack() as stack:
            shaders.glUseProgram(self._cam_frustrum_program)
            stack.callback(lambda: shaders.glUseProgram(0))

            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self._cam_frustrum_program, 'mvp'),
                1, True, mvp)

            self._cam_frustrum_positions.bind()
            position_loc = GL.glGetAttribLocation(self._cam_frustrum_program, 'in_position')
            GL.glEnableVertexAttribArray(position_loc)
            stack.callback(lambda: GL.glDisableVertexAttribArray(position_loc))
            GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                     False, 0,
                                     self._cam_frustrum_positions)
            self._cam_frustrum_positions.unbind()

            GL.glDrawArrays(GL.GL_LINES, 0, len(self._cam_frustrum_positions))


class _CameraRenderer:

    def __init__(self,
                 cam_model_files: Tuple[str, str],
                 tracked_cam_parameters: data3d.CameraParameters,
                 tracked_cam_track: List[data3d.Pose]):

        self._cam_model_renderer = _CameraModelRenderer(cam_model_files)
        self._cam_frustrum_renderer = _CameraFrustrumRenderer(tracked_cam_parameters)

        self._cam_poses = [data3d.Pose(_opencv_rotation_matrix.dot(pose.r_mat).dot(_opencv_rotation_matrix),
                                       _opencv_rotation_matrix.dot(pose.t_vec))
                           for pose in tracked_cam_track]

    def render(self, pv_matrix: np.ndarray, tracked_cam_track_pos_float: float):
        camera_pos_floor = int(np.floor(tracked_cam_track_pos_float))
        assert camera_pos_floor >= 0
        camera_pos_ceil = int(np.ceil(tracked_cam_track_pos_float))
        assert camera_pos_ceil <= len(self._cam_poses) - 1
        camera_pos_fraction = tracked_cam_track_pos_float - camera_pos_floor
        assert 0 <= camera_pos_fraction <= 1

        floor_quaternion = Quaternion(matrix=self._cam_poses[camera_pos_floor].r_mat)
        ceil_quaternion = Quaternion(matrix=self._cam_poses[camera_pos_ceil].r_mat)
        rotation_matrix = Quaternion.slerp(floor_quaternion, ceil_quaternion, camera_pos_fraction).rotation_matrix

        floor_trans_vector = self._cam_poses[camera_pos_floor].t_vec
        ceil_trans_vector = self._cam_poses[camera_pos_ceil].t_vec
        translation_vector = ceil_trans_vector * camera_pos_fraction + floor_trans_vector * (1 - camera_pos_fraction)

        # cam_model_pose_matrix = _get_pose_matrix(self._cam_poses[camera_pos_floor])
        cam_model_pose_matrix = _get_pose_matrix(data3d.Pose(r_mat=rotation_matrix, t_vec=translation_vector))
        cam_mvp = pv_matrix.dot(cam_model_pose_matrix)

        self._cam_model_renderer.render(cam_mvp)
        self._cam_frustrum_renderer.render(cam_mvp)


class CameraTrackRenderer:

    def __init__(self,
                 cam_model_files: Tuple[str, str],
                 tracked_cam_parameters: data3d.CameraParameters,
                 tracked_cam_track: List[data3d.Pose],
                 point_cloud: data3d.PointCloud):
        """
        Initialize CameraTrackRenderer. Load camera model, create buffer objects, load textures,
        compile shaders, e.t.c.
        :param cam_model_files: path to camera model obj file and texture. The model consists of
        triangles with per-point uv and normal attributes
        :param tracked_cam_parameters: tracked camera field of view and aspect ratio. To be used
        for building tracked camera frustrum
        :param point_cloud: colored point cloud
        """

        self._point_cloud_renderer = _PointCloudRenderer(point_cloud)
        self._cam_track_line_renderer = _CameraTrackLineRenderer(tracked_cam_track)
        self._cam_renderer = _CameraRenderer(cam_model_files, tracked_cam_parameters, tracked_cam_track)

        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)

    def display(self, camera_tr_vec, camera_rot_mat, camera_fov_y, tracked_cam_track_pos_float):
        """
        Draw everything with specified render camera position, projection parameters and 
        tracked camera position
        :param camera_tr_vec: vec3 position of render camera in global space
        :param camera_rot_mat: mat3 rotation matrix of render camera in global space
        :param camera_fov_y: render camera field of view. To be used for building a projection
        matrix. Use glutGet to calculate current aspect ratio
        :param tracked_cam_track_pos_float: a frame in which tracked camera
        model and frustrum should be drawn (see tracked_cam_track_pos for basic task)
        :return: returns nothing
        """

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        view_mat = np.linalg.inv(_get_pose_matrix(data3d.Pose(camera_rot_mat, camera_tr_vec)))

        aspect_ratio = float(GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH)) / float(GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT))
        focus_y = 1.0 / np.tan(camera_fov_y / 2.0)
        focus_x = focus_y / aspect_ratio

        near = 0.1
        far = 1000.0

        projection_mat = np.matrix([
            [focus_x, 0, 0, 0],
            [0, focus_y, 0, 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0],
        ])

        pv_matrix = projection_mat.dot(view_mat)

        self._point_cloud_renderer.render(pv_matrix)
        self._cam_track_line_renderer.render(pv_matrix)
        self._cam_renderer.render(pv_matrix, tracked_cam_track_pos_float)

        GLUT.glutSwapBuffers()
