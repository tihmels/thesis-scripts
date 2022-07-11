#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import re
from pathlib import Path
from shutil import copy2, rmtree

import numpy as np
from PIL import Image
from alive_progress import alive_bar
from scipy import ndimage

from VideoData import VideoData, get_frame_dir, get_frame_paths, get_shot_file, get_kf_dir, get_keyframe_paths, \
    read_shots_from_file, get_date_time
from utils.constants import TV_FILENAME_RE


def get_center_kf_idx(frames):
    return int(np.divide(len(frames), 2.0))


def get_magnitude_gradient_kf_idx(frames):
    frames = [frame.astype('int32') for frame in frames]

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

    return keyframe_idx


def detect_keyframes(vd: VideoData, kf_func=get_magnitude_gradient_kf_idx):
    segments = vd.shots

    is_summary = vd.path.parent.name == 'ts100'

    for seg_idx, (seg_start_idx, seg_end_idx) in enumerate(segments):

        frames = [Image.open(frame).convert('L') for frame in vd.frames[seg_start_idx:seg_end_idx]]
        frames = [np.array(frame) for frame in frames]

        if is_summary:
            frames = [frame[:220, :] for frame in frames]

        keyframe_idx = kf_func(frames)

        yield keyframe_idx + seg_start_idx


def check_requirements(path: Path, skip_existing=False):
    match = re.match(TV_FILENAME_RE, path.name)

    if match is None or not path.is_file():
        print(f'{path.name} does not exist or does not match TV-*.mp4 pattern.')
        return False

    frame_dir = get_frame_dir(path)

    if not frame_dir.is_dir() or not len(get_frame_paths(path)) > 0:
        print(f'{file.name} no frames have been extracted.')
        return False

    shot_file = get_shot_file(path)

    if not shot_file.is_file():
        print(f'{path.name} has no detected shots.')
        return False

    kf_dir = get_kf_dir(path)

    if skip_existing and kf_dir.is_dir() and len(get_keyframe_paths(path)) == len(
            read_shots_from_file(shot_file)):
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
    parser.add_argument('-s', '--skip', action='store_true', help="skip keyframe extraction if already exist")
    parser.add_argument('--center', action='store_true', help="use shot center frame as keyframe ")
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

        rmtree(vd.keyframe_dir, ignore_errors=True)
        vd.keyframe_dir.mkdir(parents=True)

        with alive_bar(vd.n_shots, ctrl_c=False, title=f'[{vf_idx + 1}/{len(video_files)}] {vd}', length=20) as bar:

            for kf_idx in detect_keyframes(vd, get_center_kf_idx if args.center else get_magnitude_gradient_kf_idx):
                copy2(vd.frames[kf_idx], vd.keyframe_dir)
                bar()
