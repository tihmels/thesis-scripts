#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from alive_progress import alive_bar

from common.VAO import get_date_time, VAO, get_shot_classification_file, get_shot_file, get_keyframe_dir, \
    get_keyframe_paths, read_shots_from_file
from common.constants import TV_FILENAME_RE
from shot_classifier_model.inference import classify_video_shots

parser = argparse.ArgumentParser('Video Shot Classifier')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')


def check_requirements(video: Path):
    if not re.match(TV_FILENAME_RE, video.name):
        return False

    shot_file = get_shot_file(video)

    if not shot_file.is_file():
        print(f'{video.name} has no detected shots.')
        return False

    kf_dir = get_keyframe_dir(video)

    if not kf_dir.is_dir() or not len(get_keyframe_paths(video)) == len(read_shots_from_file(shot_file)):
        print(f'{video.name} has no detected keyframes.')
        return False

    return True


def was_processed(video: Path):
    shot_file = get_shot_file(video)

    return all(shot.type for shot in read_shots_from_file(shot_file))


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    for idx, video in enumerate(video_files):
        vao = VAO(video)

        print(f'[{idx + 1}/{len(video_files)}] {vao}')

        results = []

        with alive_bar(vao.n_shots, ctrl_c=False, title=f'[{idx + 1}/{len(video_files)}] {vao}', length=20) as bar:
            for result in classify_video_shots(vao, top_n=1):
                results.append(result[0][0])
                bar()

        shots = [(sd.first_frame_idx, sd.last_frame_idx) for sd in vao.data.shots]
        shots_and_type = [(first_idx, last_idx, shot_type) for (first_idx, last_idx), shot_type in zip(shots, results)]

        df = pd.DataFrame(data=np.array(shots_and_type), columns=['first_frame_idx', 'last_frame_idx', 'type'])
        df.to_csv(get_shot_file(vao), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
