#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from VideoData import VideoData
from utils.constants import TV_FILENAME_RE
from utils.fs_utils import get_frame_dir, get_shot_file, get_date_time, get_kf_dir, read_segments_from_file, \
    print_progress_bar


def detect_keyframes(vd: VideoData):
    segments = vd.segments

    is_summary = vd.path.parent.name == 'ts100'

    for seg_idx, (seg_start_idx, seg_end_idx) in enumerate(segments):

        frames = [Image.open(frame).convert('L') for frame in vd.frames[seg_start_idx:seg_end_idx]]
        # noinspection PyTypeChecker
        frames = [np.array(frame, dtype="int32") for frame in frames]

        if is_summary:
            frames = [frame[:220, :] for frame in frames]

        gradient_magnitudes = []

        for frame in frames:
            dx = ndimage.sobel(frame, 1, mode='constant')
            dy = ndimage.sobel(frame, 0, mode='constant')

            gm = np.hypot(dx, dy)
            gradient_magnitudes.append(gm)

        def mean(gradient):
            numerator = np.sum(gradient)
            denominator = gradient.shape[0]
            return numerator / denominator

        def std(gradient, mean):
            z_grad = np.square(gradient - mean)
            numerator = np.sum(z_grad)
            denominator = gradient.shape[0]
            return np.sqrt(numerator / denominator)

        means = [mean(gm) for gm in gradient_magnitudes]
        stds = [std(gm, mean) for gm, mean in zip(gradient_magnitudes, means)]

        zgms = [np.divide(gm - mean, std) for gm, mean, std in zip(gradient_magnitudes, means, stds)]

        zgm_means = [mean(zgm) for zgm in zgms]
        zgm_stds = [std(zgm, mean) for zgm, mean in zip(zgms, zgm_means)]

        cvs = [np.divide(zgm_std, zgm_mean) for zgm_mean, zgm_std in zip(zgm_means, zgm_stds)]

        keyframe_idx = np.argmax(cvs)

        yield keyframe_idx + seg_start_idx


def check_requirements(path: Path, skip_existing: False):
    match = re.match(TV_FILENAME_RE, path.name)

    if match is None or not path.is_file():
        print(f'{path.name} does not exist or does not match TV-*.mp4 pattern.')
        return False

    frame_dir = get_frame_dir(path)

    if not frame_dir.is_dir() or not len(list(frame_dir.glob('frame_*.jpg'))) > 0:
        print(f'{file.name} no frames have been extracted.')
        return False

    shot_file = get_shot_file(path)

    if not shot_file.exists():
        print(f'{path.name} has no detected shots.')
        return False

    kf_dir = get_kf_dir(path)

    if skip_existing and kf_dir.is_dir() and len(list(kf_dir.glob('frame*.jpg'))) == len(
            read_segments_from_file(shot_file)):
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
    parser.add_argument('-s', '--skip', action='store_true', help="skip keyframe extraction if already exist")
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    video_files = []

    for file in args.files:
        if file.is_file() and check_requirements(file, args.skip):
            video_files.append(file)
        elif file.is_dir():
            video_files.extend([video for video in file.glob('*.mp4') if check_requirements(video, args.skip)])

    assert len(video_files) > 0

    video_files.sort(key=get_date_time)

    print(f'Extracting shot keyframes from {len(video_files)} videos ... \n')

    for vf_idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        # print(f'\n[{vf_idx + 1}/{len(video_files)}] {vd}')

        shutil.rmtree(vd.keyframe_dir, ignore_errors=True)
        vd.keyframe_dir.mkdir(parents=True)

        for shot_idx, kf_idx in enumerate(detect_keyframes(vd)):
            shutil.copy2(vd.frames[kf_idx], vd.keyframe_dir)

            print_progress_bar(shot_idx + 1, len(vd.segments), length=20, prefix=f'[{vf_idx + 1}/{len(video_files)}]',
                               suffix=f'{vd}')
