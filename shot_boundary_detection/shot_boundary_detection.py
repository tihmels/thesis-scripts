#!/Users/tihmels/Scripts/thesis/conda-env/bin/python

import argparse
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from numpy import asarray

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


def get_images_from_dir(directory: Path, mode: str = 'RGB', size: (int, int) = (48, 27)):
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


def subdirs(root):
    dirs = [root]
    for path in Path(root).iterdir():
        if path.is_dir():
            dirs.append(path)
            subdirs(path)
    return dirs


def process_frame_directory(directory: Path):
    print(f'\nStarting Shot Transition Detection on {directory.relative_to(directory.parent.parent)}')

    try:
        frames = get_images_from_dir(directory)
        _, scenes, img = shot_transition_detection(np.array([asarray(img) for img in frames]))

        df = pd.DataFrame(scenes)
        print(df)

        np.savetxt(Path(directory, 'scenes.txt').absolute(), scenes, fmt="%d")
        img.save(Path(directory, 'scenes.png').absolute())

        return directory

    except Exception as e:
        print(e)


def mute():
    sys.stdout = open(os.devnull, 'w')


if __name__ == "__main__":
    set_tf_loglevel(logging.FATAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', type=lambda p: Path(p).resolve(strict=True), nargs='+')
    parser.add_argument('-p', '--parallel', action='store_true')
    args = parser.parse_args()

    frame_dirs = []
    for directory in args.dirs:
        frame_dirs.extend([Path(d) for d in subdirs(directory) if len(list(d.glob('frame_*.jpg'))) > 0])

    assert len(frame_dirs) > 0, f'{args.dir} does not contain any subdirectories with frame_*.jpg files.'

    print("Executing Shot Transition Detection on...")
    [print(f'- {d}') for d in frame_dirs]

    if args.parallel:
        pool = mp.Pool(os.cpu_count(), initializer=mute)

        processes = []

        for d in frame_dirs:
            p = pool.apply_async(process_frame_directory, [d],
                                 callback=lambda d: print(f'{d.relative_to(d.parent.parent)} done'))
            processes.append(p)

        [p.get() for p in processes]

        pool.close()
        pool.join()

        print("Shutdown")

    else:
        for d in frame_dirs:
            process_frame_directory(d)
