#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import re
from pathlib import Path
from shutil import copy2

import numpy as np
from PIL import Image
from alive_progress import alive_bar
from scipy import ndimage

from common.VideoData import VideoData, get_frame_dir, get_frame_paths, get_shot_file, get_keyframe_dir, \
    get_keyframe_paths, \
    read_shots_from_file, get_date_time
from common.constants import TV_FILENAME_RE
from common.fs_utils import re_create_dir

parser = argparse.ArgumentParser('Keyframe Extraction')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help='Tagesschau video file(s)')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing',
                    help='Recalculate keyframes for all videos')
parser.add_argument('--center', action='store_true', help="Use shot center frame as keyframe")


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


def detect_keyframes(vd: VideoData, kf_func):
    shots = vd.shots

    for shot_idx, sd in enumerate(shots):

        frames = [Image.open(frame).convert('L') for frame in
                  vd.frames[sd.first_frame_idx + 5:sd.last_frame_idx - 5]]
        frames = [np.array(frame) for frame in frames]

        if vd.is_summary:
            frames = [frame[:220, :] for frame in frames]

        keyframe_idx = kf_func(frames)

        yield keyframe_idx + sd.first_frame_idx


def check_requirements(video: Path):
    if not re.match(TV_FILENAME_RE, video.name):
        return False

    frame_dir = get_frame_dir(video)

    if not frame_dir.is_dir() or not len(get_frame_paths(video)) > 0:
        print(f'{video.name} has no extracted frames.')
        return False

    shot_file = get_shot_file(video)

    if not shot_file.is_file():
        print(f'{video.name} has no detected shots.')
        return False

    return True


def was_processed(video: Path):
    kf_dir = get_keyframe_dir(video)
    shot_file = get_shot_file(video)

    return kf_dir.is_dir() and len(get_keyframe_paths(video)) == len(read_shots_from_file(shot_file))


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    print(f'Extracting shot keyframes for {len(video_files)} videos ... ', end='\n\n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        re_create_dir(vd.keyframe_dir)

        with alive_bar(vd.n_shots, ctrl_c=False, title=f'[{idx + 1}/{len(video_files)}] {vd}', length=20) as bar:
            for kf_idx in detect_keyframes(vd, get_center_kf_idx if args.center else get_magnitude_gradient_kf_idx):
                copy2(vd.frames[kf_idx], vd.keyframe_dir)
                bar()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
