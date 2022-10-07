#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common.VideoData import VideoData, get_shot_file, get_data_dir, get_frame_dir, get_frame_paths, get_date_time
from common.constants import TV_FILENAME_RE
from common.fs_utils import set_tf_loglevel

set_tf_loglevel(logging.FATAL)
from transnetv2 import TransNetV2

parser = argparse.ArgumentParser('Determine shot boundaries')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
parser.add_argument('--overwrite', action='store_true', help="Re-calculate shot boundaries for all videos")

model = TransNetV2()


def shot_transition_detection(frames):
    single_frame_predictions, all_frame_predictions = model.predict_frames(frames)

    predictions = np.stack([single_frame_predictions, all_frame_predictions], 1)
    scenes = model.predictions_to_scenes(single_frame_predictions, threshold=0.2)
    img = model.visualize_predictions(frames, (single_frame_predictions, all_frame_predictions))

    return predictions, scenes, img


def process_video(vd: VideoData):
    frames = [cv2.imread(str(frame)) for frame in vd.frames]
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    frames = [cv2.resize(frame, (48, 27)) for frame in frames]

    _, segments, img = shot_transition_detection(np.array(frames))

    data = segments[segments[:, 1] - segments[:, 0] > 10]
    data = np.c_[data, data[:, 1] - data[:, 0] + 1]

    df = pd.DataFrame(data=data, columns=['first_frame_idx', 'last_frame_idx', 'n_frames'])

    df.to_csv(get_shot_file(vd), index=False)
    img.save(Path(get_data_dir(vd), 'shots.png').absolute())

    return vd


def check_requirements(video: Path, skip_existing):
    match = re.match(TV_FILENAME_RE, video.name)

    if match is None or not video.is_file():
        return False

    frame_path = get_frame_dir(video)

    if not frame_path.is_dir() or not len(get_frame_paths(video)) > 0:
        print(f'{video.name} has no extracted frames.')
        return False

    if skip_existing and get_shot_file(video).is_file():
        return False

    return True


def mute():
    sys.stdout = open(os.devnull, 'w')


def main(args):
    video_files = []

    for file in args.files:
        if file.is_file() and check_requirements(file, not args.overwrite):
            video_files.append(file)
        elif file.is_dir():
            video_files.extend([video for video in file.glob('*.mp4') if check_requirements(video, not args.overwrite)])

    assert len(video_files) > 0

    video_files.sort(key=get_date_time)

    print(f'Video Segmentation ({len(video_files)} videos)\n')

    for idx, video in enumerate(video_files):
        vd = VideoData(video)

        print(f'[{idx + 1}/{len(video_files)}] {vd}')

        process_video(vd)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
