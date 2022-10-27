#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from alive_progress import alive_bar

from common.VideoData import get_keyframe_dir, get_date_time, VideoData, get_shot_classification_file, get_shot_file
from common.constants import TV_FILENAME_RE
from shot_classifier_model.inference import classify_video_shots

parser = argparse.ArgumentParser('Video Shot Classifier')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing', )


def check_requirements(video: Path):
    if not re.match(TV_FILENAME_RE, video.name):
        return False

    shot_file = get_shot_file(video)

    if not shot_file.is_file():
        print(f'{video.name} has no detected shots.')
        return False

    return True


def was_processed(video: Path):
    return get_shot_classification_file(video).is_file()


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    for idx, video in enumerate(video_files):
        vd = VideoData(video)

        print(f'[{idx + 1}/{len(video_files)}] {vd}')

        classifications = []

        with alive_bar(vd.n_shots, ctrl_c=False, title=f'[{idx + 1}/{len(video_files)}] {vd}', length=20) as bar:
            for classification in classify_video_shots(vd, top_n=1):
                classifications.append(classification[0][0])
                bar()

        df = pd.DataFrame(data=np.array(classifications), columns=['class'])
        df.to_csv(get_shot_classification_file(vd), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
