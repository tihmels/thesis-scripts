#!/Users/tihmels/Scripts/thesis-scripts/conda-env/bin/python

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


def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


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


def process_frame_dir(directory: Path):
    print(f'\n{directory.relative_to(directory.parent.parent)}')

    try:

        frames = get_frames_from_dir(directory)
        _, segments, img = shot_transition_detection(np.array([np.asarray(img) for img in frames]))

        segments = segments[segments[:, 1] - segments[:, 0] > 10]

        np.savetxt(Path(directory, 'shots.txt').absolute(), segments, fmt="%d")
        img.save(Path(directory, 'shots.png').absolute())

        return directory

    except Exception as e:
        print(e)
        return e


def check_requirements(path: Path, skip_existing=False):
    match = re.match(r'^(\d{4})$', path.name)

    if match is None or not path.exists() or not path.is_dir():
        return False

    if len(list(path.glob('frame_*.jpg'))) == 0:
        print(f'{path} has no extracted frames.')
        return False

    if skip_existing and Path(path, 'shots.txt').exists():
        print(f'{path} has already shots detected. Skip ...')
        return False

    return True


def subdirs(root: str):
    sub_folders = [f.path for f in os.scandir(root) if f.is_dir()]
    for dir_name in list(sub_folders):
        sub_folders.extend(subdirs(dir_name))
    return [Path(f) for f in sub_folders]


def mute():
    sys.stdout = open(os.devnull, 'w')


if __name__ == "__main__":
    set_tf_loglevel(logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', type=lambda p: Path(p).resolve(strict=True), nargs='+')
    parser.add_argument('-s', '--skip', action='store_true', help="skip sbd if already exist")
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    frame_dirs = []
    for directory in args.dirs:
        frame_dirs.extend([d for d in sorted(subdirs(directory)) if check_requirements(d, args.skip)])

    assert len(frame_dirs) > 0, f'{args.dirs} does not contain any subdirectories with frame_*.jpg files.'

    print(f'\nVideo Segmentation ({len(frame_dirs)} videos)')


    def callback_handler(result):
        if result is not None and isinstance(result, Path):
            print(f'{result.relative_to(result.parent.parent)} done')


    if args.parallel:

        with mp.Pool(os.cpu_count(), initializer=mute) as pool:
            [pool.apply_async(process_frame_dir, (d,), callback=callback_handler) for d in frame_dirs]

            pool.close()
            pool.join()

    else:
        for d in frame_dirs:
            result = process_frame_dir(d)
            callback_handler(result)
