#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

from common.VideoData import VideoData, get_date_time, get_shot_file

parser = argparse.ArgumentParser()
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help="Tagesschau video file(s)")
parser.add_argument('--overwrite', action='store_false', dest='skip_existing',
                    help="Re-calculate shot boundaries for all videos")


def get_start_frame(vd: VideoData):
    transcripts = vd.transcript

    for idx, td in enumerate(transcripts):

        if fuzz.partial_ratio(td.text, 'Heute im Studio:') > 90:
            intro_text_seconds = transcripts[idx + 2].start.second
            first_frame_idx = intro_text_seconds * 25

            shots = np.array([(shot.first_frame_idx, shot.last_frame_idx) for shot in vd.shots])

            if np.logical_and(first_frame_idx - 10 < shots[:, 0], shots[:, 0] < first_frame_idx + 10).any():
                break

            after_index = np.argmax(shots[:, 0] > first_frame_idx)

            before_shot = shots[after_index - 1]
            after_shot = shots[after_index]

            before_shot_new = (before_shot[0], first_frame_idx - 1)
            new_shot = (first_frame_idx, after_shot[0] - 1)

            shots[after_index - 1] = before_shot_new
            new_shots = np.insert(shots, after_index, new_shot, axis=0)

            df = pd.DataFrame(data=new_shots, columns=['first_frame_idx', 'last_frame_idx'])

            df.to_csv(get_shot_file(vd), index=False)




def check_requirements(video: Path):
    return True


def was_processed(video: Path):
    return False


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    print(f'Detecting shot boundaries for {len(video_files)} videos ...', end='\n\n')

    for vf_idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        get_start_frame(vd)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
