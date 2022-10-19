#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common.VideoData import VideoData, get_shot_file, get_data_dir, get_frame_dir, get_frame_paths, get_date_time
from common.constants import TV_FILENAME_RE
from common.fs_utils import set_tf_loglevel

set_tf_loglevel(logging.FATAL)
from transnetv2 import TransNetV2

parser = argparse.ArgumentParser('Shot Boundary Detection (SBD) using TransNet V2')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help="Tagesschau video file(s)")
parser.add_argument('--threshold', type=float, default=0.2, help='Threshold for scene prediction')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing',
                    help="Re-calculate shot boundaries for all videos")

model = TransNetV2()


def shot_transition_detection(frames, threshold=0.2):
    single_frame_predictions, all_frame_predictions = model.predict_frames(frames)

    predictions = np.stack([single_frame_predictions, all_frame_predictions], 1)
    scenes = model.predictions_to_scenes(single_frame_predictions, threshold=threshold)
    img = model.visualize_predictions(frames, (single_frame_predictions, all_frame_predictions))

    return predictions, scenes, img


def detect_shot_boundaries(vd: VideoData, threshold):
    frames = [cv2.imread(str(frame)) for frame in vd.frames]
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    frames = [cv2.resize(frame, (48, 27)) for frame in frames]

    _, segments, img = shot_transition_detection(np.array(frames), threshold)

    segments = segments[segments[:, 1] - segments[:, 0] > 10]

    return segments, img


def check_requirements(video: Path):
    if not re.match(TV_FILENAME_RE, video.name):
        return False

    frame_path = get_frame_dir(video)

    if not frame_path.is_dir() or not len(get_frame_paths(video)) > 0:
        print(f'{video.name} has no extracted frames.')
        return False

    return True


def was_processed(video: Path):
    return get_shot_file(video).is_file()


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    print(f'Detecting shot boundaries for {len(video_files)} videos ...', end='\n\n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vd}')

        segments, img = detect_shot_boundaries(vd, args.threshold)

        df = pd.DataFrame(data=segments, columns=['first_frame_idx', 'last_frame_idx'])

        df.to_csv(get_shot_file(vd), index=False)
        img.save(Path(get_data_dir(vd), 'shots.png').absolute())


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
