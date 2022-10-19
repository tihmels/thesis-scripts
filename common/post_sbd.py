#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
from pathlib import Path

import numpy as np
from fuzzywuzzy import fuzz

from common.VideoData import VideoData, get_date_time, read_transcript_from_file, get_transcript_file

parser = argparse.ArgumentParser()
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help="Tagesschau video file(s)")
parser.add_argument('--overwrite', action='store_false', dest='skip_existing',
                    help="Re-calculate shot boundaries for all videos")


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

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        transcripts = read_transcript_from_file(get_transcript_file(vd))

        for t_idx, (start, end, caption) in enumerate(transcripts):

            if fuzz.partial_ratio(caption, 'Guten Abend, ich begrüße Sie zur tagesschau') > 90:
                first_text = t_idx + 1
                first_text_seconds = transcripts[first_text][0].second

                first_frame_idx = first_text_seconds * 25

                shots = vd.shots


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
