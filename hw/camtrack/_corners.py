__all__ = [
    'FrameCorners',
    'filter_frame_corners',
    'CornerStorage',
    'StorageImpl',
    'StorageFilter',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks',
    'create_cli'
]

import abc
import pickle
from typing import IO

import click
import cv2
import numpy as np

import frameseq


class FrameCorners:
    """
    namedtuple-like class representing corners belonging to one frame.

    All fields should be NumPy 2D arrays of shape=(-1, 2) or shape=(-1, 1).

    All data should be sorted by corner ids to allow usage of binary search
    (np.searchsorted).
    """

    __slots__ = ('_ids', '_points', '_sizes')

    def __init__(self, ids, points, sizes):
        """
        Construct FrameCorners.

        You may add your own fields if needed.

        :param ids: integer ids of corners
        :param points: coordinates of corners
        :param sizes: block sizes used for corner calculation (in pixels on original image format)
        """
        sorting_idx = np.argsort(ids.flatten())
        self._ids = ids[sorting_idx].reshape(-1, 1)
        self._points = points[sorting_idx].reshape(-1, 2)
        self._sizes = sizes[sorting_idx].reshape(-1, 1)

    @property
    def ids(self):
        return self._ids

    @property
    def points(self):
        return self._points

    @property
    def sizes(self):
        return self._sizes

    def __iter__(self):
        yield self.ids
        yield self.points
        yield self.sizes


def filter_frame_corners(frame_corners: FrameCorners,
                         mask: np.ndarray) -> FrameCorners:
    """
    Filter frame corners using boolean mask.

    :param frame_corners: frame corners to filter.
    :param mask: boolean mask, all elements marked by False will be filtered out.
    :return: filtered corners.
    """
    return FrameCorners(*[field[mask] for field in frame_corners])


def _to_int_tuple(point):
    return tuple(map(int, np.round(point)))


class _ColorGenerator:

    def __init__(self):
        self._rng = np.random.RandomState()

    def __call__(self, corner_id):
        self._rng.seed(corner_id)
        return _to_int_tuple(self._rng.random(size=(3,)))


def draw(grayscale_image: np.ndarray, corners: FrameCorners) -> np.ndarray:
    """
    Draw corners on image.

    :param grayscale_image: grayscale float32 image.
    :param corners: corners to draw, pyramid levels must be less than 7.
    :return: BGR image with drawn corners.
    """
    bgr = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
    colors = map(_ColorGenerator(), corners.ids)
    for color, point, block_size in zip(colors, corners.points, corners.sizes):
        point = _to_int_tuple(point)
        radius = int(block_size / 2)
        cv2.circle(bgr, point, radius, color)
    return bgr


class CornerStorage(abc.ABC):
    """
    Base class for corner storage. Acts like simple Python list.
    """

    @abc.abstractmethod
    def __getitem__(self, frame: int) -> FrameCorners:
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def max_corner_id(self):
        pass


class StorageImpl(CornerStorage):
    """
    Corner storage implementation
    """

    def __init__(self, corners_for_each_frame):
        """
        Constructor. For internal use only.
        """
        super().__init__()
        self._corners = list(corners_for_each_frame)
        self._max_id = max(c.ids.max() for c in self._corners)

    def __getitem__(self, frame: int) -> FrameCorners:
        return self._corners[frame]

    def __len__(self):
        return len(self._corners)

    def __iter__(self):
        return iter(self._corners)

    def max_corner_id(self):
        return self._max_id


class StorageFilter(CornerStorage):
    """
    Corners filterer.
    """

    def __init__(self, corner_storage, predicate):
        """
        Constructor. For internal use only.
        """
        super().__init__()
        self._storage = corner_storage
        self._predicate = predicate

    def __getitem__(self, frame: int) -> FrameCorners:
        frame_corners = self._storage[frame]
        mask = self._predicate(frame_corners)
        return filter_frame_corners(frame_corners, mask)

    def __len__(self):
        return len(self._storage)

    def __iter__(self):
        for frame in range(len(self)):  # pylint:disable=consider-using-enumerate
            yield self[frame]

    def max_corner_id(self):
        return self._storage.max_corner_id()


def calc_track_interval_mappings(corner_storage: CornerStorage) -> np.ndarray:
    max_id = max(corners.ids.max() for corners in corner_storage)
    left = np.full((max_id + 1,), len(corner_storage))
    right = np.full((max_id + 1,), -1)
    for i, corners in enumerate(corner_storage):
        unique = np.unique(corners.ids)
        left[unique] = np.minimum(left[unique], i)
        right[unique] = np.maximum(right[unique], i)
    return left, right


def calc_track_len_array_mapping(corner_storage: CornerStorage) -> np.ndarray:
    """
    Calculate lengths of all tracks in the given corner storage.

    :param corner_storage: corner storage to calculate track lengths.
    :return: ndarray, i-th element contains the length of the track with id=i.
    """
    left, right = calc_track_interval_mappings(corner_storage)
    mask = left <= right
    counter = np.zeros_like(left)
    counter[mask] = right[mask] - left[mask] + 1
    return counter


def without_short_tracks(corner_storage: CornerStorage,
                         min_len: int) -> CornerStorage:
    """
    Create corner storage wrapper to filter out short tracks.

    :param corner_storage: storage to wrap.
    :param min_len: min allowed track length.
    :return: filtered corner storage.
    """
    counter = calc_track_len_array_mapping(corner_storage)

    def predicate(corners):
        return counter[corners.ids.flatten()] >= min_len

    return StorageFilter(corner_storage, predicate)


def dump(corner_storage: CornerStorage, stream: IO[bytes]) -> None:
    """
    Dump corner storage.

    :param stream: file-like writable object.
    """
    pickle.dump(list(corner_storage), stream)


def load(stream: IO[bytes]) -> CornerStorage:
    """
    Load corner storage.

    :param stream: file-like readable object.
    :return: loaded corner storage.
    """
    return StorageImpl(pickle.load(stream))


def create_cli(build):
    """
    Create command line interface for 'corners' module.

    :param build: function that builds corner storage from frame sequence.
    :return: CLI function.
    """

    @click.command()
    @click.argument('frame_sequence')
    @click.option('file_to_load', '--load-corners', type=click.File('rb'))
    @click.option('file_to_dump', '--dump-corners', type=click.File('wb'))
    @click.option('--show', is_flag=True)
    def cli(frame_sequence, file_to_load, file_to_dump, show):
        """
        FRAME_SEQUENCE path to a video file or shell-like wildcard describing
        multiple images
        """
        sequence = frameseq.read_grayscale_f32(frame_sequence)
        if file_to_load is not None:
            corner_storage = load(file_to_load)
        else:
            corner_storage = build(sequence)
        if file_to_dump is not None:
            dump(corner_storage, file_to_dump)
        if show:
            click.echo(
                "Press 'q' to stop, 'd' to go forward, 'a' to go backward, "
                "'r' to reset"
            )
            frame = 0
            while True:
                grayscale = sequence[frame]
                bgr = draw(grayscale, corner_storage[frame])
                cv2.imshow('Frame', bgr)
                key = chr(cv2.waitKey(20) & 0xFF)
                if key == 'a' and frame > 0:
                    frame -= 1
                if key == 'd' and frame + 1 < len(corner_storage):
                    frame += 1
                if key == 'q':
                    break
                if key == 'r':
                    frame = 0

    return cli
