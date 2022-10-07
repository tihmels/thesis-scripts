#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from alive_progress import alive_bar

from common.VideoData import get_keyframe_dir, get_date_time, VideoData, get_shot_type_file
from common.constants import TV_FILENAME_RE
from shot_classifier_model.inference import classify_video_shots

parser = argparse.ArgumentParser('Determine shot boundaries')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
parser.add_argument('--overwrite', action='store_true', help="Re-calculate shot boundaries for all videos")


def check_requirements(path: Path, skip_existing: bool):
    match = re.match(TV_FILENAME_RE, path.name)

    if match is None or not path.is_file():
        return False

    kf_dir = get_keyframe_dir(path)

    if not kf_dir.is_dir() or len(list(kf_dir.glob("*.jpg"))) < 1:
        return False

    return True


def main(args):
    video_files = []

    for file in args.files:
        if file.is_file() and check_requirements(file, not args.overwrite):
            video_files.append(file)
        elif file.is_dir():
            video_files.extend([video for video in file.glob('*.mp4') if check_requirements(video, not args.overwrite)])

    assert len(video_files) > 0

    video_files.sort(key=get_date_time)

    for idx, video in enumerate(video_files):
        vd = VideoData(video)

        types = []

        with alive_bar(vd.n_shots, ctrl_c=False, title=f'[{idx + 1}/{len(video_files)}] {vd}', length=20) as bar:
            for result in classify_video_shots(vd, top_n=1):
                types.append(result[0][0])
                bar()

        df = pd.DataFrame(data=np.array(types), columns=['type'])
        df.to_csv(get_shot_type_file(vd), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
