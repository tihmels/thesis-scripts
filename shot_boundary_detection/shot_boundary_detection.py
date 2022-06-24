#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python

import argparse
import logging
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from transnetv2 import TransNetV2
from utils.constants import TV_FILENAME_RE
from utils.fs_utils import get_frame_dir, get_shot_file, get_data_dir, set_tf_loglevel


def get_frames_from_dir(directory: Path, mode: str = 'RGB', size: (int, int) = (48, 27)):
    assert directory.is_dir(), f'{directory} is not a directory'
    assert len(list(directory.glob('frame_*.jpg'))) > 0, f'{directory} does not contain any frame_*.jpg files'

    frames = sorted(directory.glob('frame_*.jpg'))
    return [Image.open(f).convert(mode).resize(size) for f in frames]


def shot_transition_detection(frames):
    model = TransNetV2()

    single_frame_predictions, all_frame_predictions = model.predict_frames(frames)

    predictions = np.stack([single_frame_predictions, all_frame_predictions], 1)
    scenes = model.predictions_to_scenes(single_frame_predictions)
    img = model.visualize_predictions(frames, (single_frame_predictions, all_frame_predictions))

    return predictions, scenes, img


def process_video(video: Path):
    frame_dir = get_frame_dir(video)

    try:

        frames = get_frames_from_dir(frame_dir)
        _, segments, img = shot_transition_detection(np.array([np.asarray(img) for img in frames]))

        segments = segments[segments[:, 1] - segments[:, 0] > 10]

        np.savetxt(get_shot_file(video).absolute(), segments, fmt="%d")
        img.save(Path(get_data_dir(video), 'shots.png').absolute())

        return video

    except Exception as e:
        print(e)
        return e


def check_requirements(video: Path, skip_existing: bool):
    match = re.match(TV_FILENAME_RE, video.name)

    if match is None or not video.is_file():
        return False

    frame_path = get_frame_dir(video)

    if not frame_path.exists() or len(list(frame_path.glob('frame_*.jpg'))) == 0:
        print(f'{video.name} has no extracted frames.')
        return False

    if skip_existing and get_shot_file(video).exists():
        print(f'{video.name} has already shots detected. Skip ...')
        return False

    return True


def mute():
    sys.stdout = open(os.devnull, 'w')


if __name__ == "__main__":
    set_tf_loglevel(logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=lambda p: Path(p).resolve(strict=True))
    parser.add_argument('-s', '--skip', action='store_true', help="skip sbd if shots.txt already exist")
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    video_files = [file for file in sorted(args.dir.glob('*.mp4')) if check_requirements(file, args.skip)]

    assert len(video_files) > 0

    print(f'\nVideo Segmentation ({len(video_files)} videos)')


    def callback_handler(res):
        if res is not None and isinstance(res, Path):
            print(f'{res.relative_to(res.parent.parent)} done')


    if args.parallel:

        with mp.Pool(os.cpu_count(), initializer=mute) as pool:
            [pool.apply_async(process_video, (video,), callback=callback_handler) for video in video_files]

            pool.close()
            pool.join()

    else:
        for video in video_files:
            result = process_video(video)
            callback_handler(result)
