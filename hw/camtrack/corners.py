#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


_QUALITY = 0.05
_MIN_DIST = 15
_CORNER_SIZE = 10
_PYR_SIZE = 3
_WINDOW_SIZE = _CORNER_SIZE * (2 ** _PYR_SIZE)


def configure(shape):
    global _CORNER_SIZE
    global _MIN_DIST
    global _WINDOW_SIZE
    shape = min(shape[0], shape[1])

    r = _MIN_DIST / _CORNER_SIZE
    _CORNER_SIZE = _CORNER_SIZE * shape // 1000
    _MIN_DIST = int(r * _CORNER_SIZE)
    _WINDOW_SIZE = _CORNER_SIZE * (2 ** _PYR_SIZE)


def _detect_corners_pyr(image, ids, corners, sizes):
    mask = np.full_like(image, 255, dtype='uint8')
    if corners is not None:
        order = sorted(range(len(ids)), key=lambda x: sizes[x][0])
        cur_corners = corners
        cur_sizes = sizes
        cur_ids = ids

        corners = []
        sizes = []
        ids = []

        for i in order:
            corner = cur_corners[i].astype(int)
            if corner[1] < 0 or corner[1] >= mask.shape[0] or corner[0] < 0 or corner[0] >= mask.shape[1]:
                continue
            if mask[corner[1], corner[0]] == 0:
                continue
            cv2.circle(mask, corner, int(_MIN_DIST * cur_sizes[i][0] / _CORNER_SIZE), 0, -1)
            corners.append(cur_corners[i])
            sizes.append(cur_sizes[i][0])
            ids.append(cur_ids[i][0])

        corners = np.array(corners).reshape([-1, 2])
        sizes = np.array(sizes).reshape([-1, 1])
        ids = np.array(ids).reshape([-1, 1])

    orig_size = image.shape[0]

    ext_corners = []
    cur_img = image.copy()
    cur_mask = mask.copy()

    for _ in range(_PYR_SIZE):
        cur_corners, cur_quality = cv2.goodFeaturesToTrackWithQuality(
            image=cur_img,
            mask=cur_mask,
            maxCorners=0,
            qualityLevel=_QUALITY,
            minDistance=_MIN_DIST,
            blockSize=_CORNER_SIZE
        )
        if cur_corners is not None:
            cur_corners = cur_corners * orig_size / cur_img.shape[0]
            cur_size = int(_CORNER_SIZE * orig_size / cur_img.shape[0])

            ext_corners.extend(zip(cur_corners, [cur_size] * len(cur_corners), cur_quality))

        cur_img = cv2.pyrDown(cur_img)
        cur_mask = cv2.pyrDown(cur_mask)

    sorted_ext = sorted(ext_corners, key=lambda x: (-x[2][0], x[1]))

    next_id = 0 if ids is None else ids.max() + 1
    ext_ids = []
    ext_corners = []
    ext_sizes = []

    for corner, size, _ in sorted_ext:
        corner = corner[0]
        if mask[int(corner[1]), int(corner[0])] == 0:
            continue
        cv2.circle(mask, corner.astype(int), int(_MIN_DIST * size / _CORNER_SIZE), 0, -1)
        ext_ids.append(next_id)
        ext_corners.append(corner)
        ext_sizes.append(size)
        next_id += 1

    if not ext_ids:
        return ids, corners, sizes

    ext_ids = np.array(ext_ids).reshape([-1, 1])
    ext_corners = np.array(ext_corners)
    ext_sizes = np.array(ext_sizes).reshape([-1, 1])

    if ids is None:
        ids = ext_ids
        corners = ext_corners
        sizes = ext_sizes
    else:
        ids = np.vstack([ids, ext_ids])
        corners = np.vstack([corners, ext_corners])
        sizes = np.vstack([sizes, ext_sizes])

    return ids, corners, sizes


def _detect_flow(image0, image1, ids, corners, sizes):
    def to_uint8(img):
        return (img * 255).astype('uint8')

    next_corners, st, err = cv2.calcOpticalFlowPyrLK(
        to_uint8(image0), to_uint8(image1), corners.astype('float32'), None,
        winSize=[_WINDOW_SIZE] * 2, maxLevel=_PYR_SIZE
    )

    st = st.flatten() == 1

    ids = ids[st]
    next_corners = next_corners[st]
    sizes = sizes[st]

    if len(ids) == 0:
        return None, None, None

    return ids, next_corners, sizes


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    configure(frame_sequence[0].shape)
    ids, corners, sizes = _detect_corners_pyr(frame_sequence[0], None, None, None)

    for frame, image in enumerate(frame_sequence):
        if frame > 0:
            ids, corners, sizes = _detect_flow(frame_sequence[frame - 1], image, ids, corners, sizes)
        ids, corners, sizes = _detect_corners_pyr(image, ids, corners, sizes)

        builder.set_corners_at_frame(frame, FrameCorners(ids.astype('int64'), corners, sizes))


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
