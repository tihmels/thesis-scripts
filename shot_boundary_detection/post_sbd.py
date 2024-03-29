#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

from common.VAO import VAO, get_date_time, is_summary, get_main_transcript_file, get_shot_file
from common.constants import TV_FILENAME_RE
from common.utils import sec_to_frame_idx

parser = argparse.ArgumentParser()
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help="Tagesschau video file(s)")
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')

welcoming = "(Willkommen|Guten Abend) meine Damen und Herren, ich begrüße sie zur Tagesschau"


def fix_first_anchorshot_segment(shots, transcript):
    for idx, td in enumerate(transcript):

        if fuzz.token_set_ratio(td.text, welcoming) > 90:
            first_news_sentence = transcript[idx + 1]
            first_news_frame_idx = sec_to_frame_idx(first_news_sentence.start.second)

            if np.logical_and(first_news_frame_idx - 5 < shots[:, 0],
                              shots[:, 0] < first_news_frame_idx + 5).any():
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
        vao = VAO(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vao}')

        shots = np.array([(shot.first_frame_idx, shot.last_frame_idx) for shot in vao.data.shots])

        segments = fix_first_anchorshot_segment(shots, vao.data.transcripts)

        df = pd.DataFrame(data=segments, columns=['first_frame_idx', 'last_frame_idx'])

        df.to_csv(get_shot_file(vao), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
