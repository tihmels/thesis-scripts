#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

from common.VideoData import VideoData, get_date_time, is_summary, get_main_transcript_file, get_shot_file
from common.constants import TV_FILENAME_RE
from common.fs_utils import sec_to_frame_idx

parser = argparse.ArgumentParser()
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help="Tagesschau video file(s)")
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')

welcoming_1 = "ich begrüße Sie zur tagesschau"
welcoming_2 = "Willkommen zur tagesschau"


def fix_first_anchorshot_segment(vd: VideoData, shots):
    transcript = vd.transcripts

    for idx, td in enumerate(transcript):

        if fuzz.token_set_ratio(td.text, welcoming_1) > 90 or fuzz.token_set_ratio(td.text, welcoming_2) > 90:
            first_news_sentence = transcript[idx + 1]
            first_news_frame_idx = sec_to_frame_idx(first_news_sentence.start.second)

            if np.logical_and(first_news_frame_idx - 10 < shots[:, 0],
                              shots[:, 0] < first_news_frame_idx + 10).any():
                break

            after_shot_idx = np.argmax(shots[:, 0] > first_news_frame_idx)

            after_shot = shots[after_shot_idx]
            before_shot = shots[after_shot_idx - 1]

            updated_before_shot = (before_shot[0], first_news_frame_idx - 1)
            new_shot = (first_news_frame_idx, after_shot[0] - 1)

            shots[after_shot_idx - 1] = updated_before_shot
            updated_shots = np.insert(shots, after_shot_idx, new_shot, axis=0)

            return updated_shots


def check_requirements(video: Path):
    if not re.match(TV_FILENAME_RE, video.name):
        return False

    if is_summary(video):
        return False

    shot_file = get_shot_file(video)

    if not shot_file.is_file():
        return False

    transcript_file = get_main_transcript_file(video)

    if not transcript_file.is_file():
        return False

    return True


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    print(f'Detecting shot boundaries for {len(video_files)} videos ...', end='\n\n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vd}')

        transcript = vd.transcripts
        shots = np.array([(shot.first_frame_idx, shot.last_frame_idx) for shot in vd.shots])

        segments = fix_first_anchorshot_segment(transcript, shots)

        df = pd.DataFrame(data=segments, columns=['first_frame_idx', 'last_frame_idx'])

        df.to_csv(get_shot_file(vd), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
