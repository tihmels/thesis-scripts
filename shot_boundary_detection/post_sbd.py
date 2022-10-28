#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

from common.VideoData import VideoData, get_date_time, is_summary, get_main_transcript_file, get_shot_file
from common.constants import TV_FILENAME_RE

parser = argparse.ArgumentParser()
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help="Tagesschau video file(s)")
parser.add_argument('--overwrite', action='store_false', dest='skip_existing',
                    help="Re-calculate shot boundaries for all videos")


def fix_first_anchorshot_segment(transcript, segments):
    for idx, td in enumerate(transcript):
        if fuzz.partial_ratio(td.text, 'Heute im Studio:') > 90:
            first_story_sentence = transcript[idx + 2]
            first_story_sentence_frame_idx = first_story_sentence.start.second * 25

            if np.logical_and(first_story_sentence_frame_idx - 10 < segments[:, 0],
                              segments[:, 0] < first_story_sentence_frame_idx + 10).any():
                break

            after_index = np.argmax(segments[:, 0] > first_story_sentence_frame_idx)

            before_shot = segments[after_index - 1]
            after_shot = segments[after_index]

            before_shot_new = (before_shot[0], first_story_sentence_frame_idx - 1)
            new_shot = (first_story_sentence_frame_idx, after_shot[0] - 1)

            segments[after_index - 1] = before_shot_new
            new_shots = np.insert(segments, after_index, new_shot, axis=0)

            return new_shots


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

        transcript = vd.transcript
        shots = np.array([(shot.first_frame_idx, shot.last_frame_idx) for shot in vd.shots])

        segments = fix_first_anchorshot_segment(transcript, shots)

        df = pd.DataFrame(data=segments, columns=['first_frame_idx', 'last_frame_idx'])

        df.to_csv(get_shot_file(vd), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
