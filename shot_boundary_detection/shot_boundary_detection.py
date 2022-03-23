#!/Users/tihmels/Scripts/thesis/conda-env/bin/python

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from numpy import asarray

from transnetv2 import TransNetV2


def load_frames_from_dir(frame_dir: Path):
    if not frame_dir.is_dir() or len(list(frame_dir.glob('frame_*.jpg'))) == 0:
        raise Exception(f'{frame_dir} is not a directory or does not contain any .jpg files')

    frames = sorted(frame_dir.glob('frame_*.jpg'))
    return np.array([asarray(Image.open(f).convert('RGB').resize((48, 27))) for f in frames], dtype=np.uint8)


def predict_shot_boundaries(frames, visualize=False):
    model = TransNetV2()

    single_frame_predictions, all_frame_predictions = model.predict_frames(frames)

    scenes = model.predictions_to_scenes(single_frame_predictions)

    predictions = np.stack([single_frame_predictions, all_frame_predictions], 1)
    np.savetxt(Path(frame_dir, 'predictions.txt').absolute(), predictions, fmt="%.6f")
    np.savetxt(Path(frame_dir, 'scenes.txt').absolute(), scenes, fmt="%d")

    if visualize:
        pil_img = model.visualize_predictions(frames,
                                              predictions=(single_frame_predictions, all_frame_predictions))
        pil_img.save(Path(frame_dir, 'vis.png').absolute())


def subdirs(root):
    dirs = []
    for path in Path(root).iterdir():
        if path.is_dir():
            dirs.append(path)
            subdirs(path)
    return dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', type=lambda p: Path(p).resolve(strict=True), nargs='+')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    frame_dirs = []
    for directory in args.dirs:
        frame_dirs.extend([Path(d) for d in subdirs(directory) if len(list(d.glob('frame_*.jpg'))) > 0])

    assert len(frame_dirs) > 0, f'{args.dir} does not contain any subdirectories with *.jpg files.'

    print(f'Shot Boundary Detection')
    [print(f'- {d}') for d in frame_dirs]

    for frame_dir in frame_dirs:
        print(f'\nEstimating shot boundaries for {frame_dir.relative_to(frame_dir.parent.parent)}')

        try:
            video_frames = load_frames_from_dir(frame_dir)
            predict_shot_boundaries(video_frames, args.visualize)
        except Exception as e:
            print(e)
