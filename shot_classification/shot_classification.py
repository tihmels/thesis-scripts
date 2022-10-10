#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from alive_progress import alive_bar

from common.VideoData import get_keyframe_dir, get_date_time, VideoData, get_shot_type_file, get_shot_file
from common.constants import TV_FILENAME_RE
from shot_classifier_model.inference import classify_video_shots

parser = argparse.ArgumentParser('Video Shot Classifier')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
parser.add_argument('--overwrite', action='store_true')


def check_requirements(path: Path, skip_existing: bool):
    match = re.match(TV_FILENAME_RE, path.name)

    if match is None or not path.is_file():
        return False

    shot_file = get_shot_file(path)

    if not shot_file.is_file():
        print(f'{path.name} has no detected shots.')
        return False

    kf_dir = get_keyframe_dir(path)

    # if not kf_dir.is_dir() or len(list(kf_dir.glob("*.jpg"))) != len(read_shots_from_file(shot_file)):
    #    return False

    return True


def main(args):
    video_files = set()

    for file in args.files:
        if file.is_file() and check_requirements(file, not args.overwrite):
            video_files.add(file)
        elif file.is_dir():
            video_files.add([video for video in file.glob('*.mp4') if check_requirements(video, not args.overwrite)])

    assert len(video_files) > 0

    video_files = sorted(video_files, key=get_date_time)

    for idx, video in enumerate(video_files):
        vd = VideoData(video)

        classifications = []

        with alive_bar(vd.n_shots, ctrl_c=False, title=f'[{idx + 1}/{len(video_files)}] {vd}', length=20) as bar:
            for classification in classify_video_shots(vd, top_n=1):
                classifications.append(classification[0][0])
                bar()

        df = pd.DataFrame(data=np.array(classifications), columns=['class'])
        df.to_csv(get_shot_type_file(vd), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
