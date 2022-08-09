#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import logging
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from VideoData import VideoData, get_shot_file, get_data_dir, get_frame_dir, get_frame_paths, get_date_time
from transnetv2 import TransNetV2
from utils.constants import TV_FILENAME_RE
from utils.fs_utils import set_tf_loglevel


def shot_transition_detection(frames):
    model = TransNetV2()

    single_frame_predictions, all_frame_predictions = model.predict_frames(frames)

    predictions = np.stack([single_frame_predictions, all_frame_predictions], 1)
    scenes = model.predictions_to_scenes(single_frame_predictions)
    img = model.visualize_predictions(frames, (single_frame_predictions, all_frame_predictions))

    return predictions, scenes, img


def process_video(vd: VideoData):
    try:

        frames = [Image.open(frame).resize((48, 27)) for frame in vd.frames]
        frames = [np.array(frame) for frame in frames]

        if vd.is_summary:
            frames = [frame[:220, :] for frame in frames]

        _, segments, img = shot_transition_detection(np.array(frames))

        data = segments[segments[:, 1] - segments[:, 0] > 10]
        data = np.c_[data, data[:, 1] - data[:, 0] + 1]

        df = pd.DataFrame(data=data, columns=['first_frame_idx', 'last_frame_idx', 'n_frames'])
        df.index = df.index + 1

        return video

    except Exception as e:
        print(e)
        return e


def check_requirements(video: Path, skip_existing: bool):
    match = re.match(TV_FILENAME_RE, video.name)

    if match is None or not video.is_file():
        return False

    frame_path = get_frame_dir(video)

    if not frame_path.is_dir() or not len(get_frame_paths(video)) > 0:
        print(f'{video.name} has no extracted frames.')
        return False

    if skip_existing and get_shot_file(video).is_file():
        # print(f'{video.name} has already shots detected. Skip ...')
        return False

    return True


def mute():
    sys.stdout = open(os.devnull, 'w')


if __name__ == "__main__":
    set_tf_loglevel(logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
    parser.add_argument('-s', '--skip', action='store_true', help="skip sbd if shots.txt already exist")
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

    print(f'Video Segmentation ({len(video_files)} videos)\n')


    def callback_handler(res):
        if res is not None and isinstance(res, Path):
            print(f'{res.relative_to(res.parent.parent)} done')


    if args.parallel:

        with mp.Pool(os.cpu_count(), initializer=mute) as pool:
            [pool.apply_async(process_video, (VideoData(vf),), callback=callback_handler) for vf in video_files]

            pool.close()
            pool.join()

    else:
        for idx, video in enumerate(video_files):
            vd = VideoData(video)

            print(f'[{idx + 1}/{len(video_files)}] {vd}')

            process_video(vd)
            print()
