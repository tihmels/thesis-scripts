#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import re
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path

from common.VideoData import get_date_time, VideoData, get_main_transcript_file, get_story_file, get_story_transcripts, \
    read_scenes_from_file
from common.constants import TV_FILENAME_RE

parser = ArgumentParser('Automatic Speech Recognition')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help='Tagesschau video file(s)')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')


def sec_to_time(seconds):
    return (datetime.min + timedelta(seconds=seconds)).time()


def split_story_transcripts(vd: VideoData):
    stories = vd.scenes

    transcript = vd.transcript

    for first_frame_idx, last_frame_idx in stories[['first_frame_idx', 'last_frame_idx']].itertuples(index=False):
        start = sec_to_time(first_frame_idx / 25)
        end = sec_to_time(last_frame_idx / 25)

        start = start.replace(microsecond=0)
        end = end.replace(second=end.second + 1, microsecond=0)

        yield list(filter(lambda t: start <= t.start and t.end <= end, transcript))


def check_requirements(video: Path):
    if not re.match(TV_FILENAME_RE, video.name):
        return False

    trans_file = get_main_transcript_file(video)

    if not trans_file.is_file():
        return False

    story_file = get_story_file(video)

    if not story_file.is_file():
        return False

    return True


def was_processed(video: Path):
    story_transcripts = get_story_transcripts(video)
    stories = read_scenes_from_file(get_story_file(video))

    return len(story_transcripts) == len(stories)


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    print(f'Split audio of {len(video_files)} videos ... \n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vd}')

        vd.transcripts_dir.mkdir(exist_ok=True)

        for story_idx, transcript in enumerate(split_story_transcripts(vd)):
            text = ' '.join([transcript.text for transcript in transcript])

            target_file = Path(vd.transcripts_dir, 'story_' + str(story_idx) + '.txt')

            with open(target_file, 'w') as file:
                file.write(text)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
