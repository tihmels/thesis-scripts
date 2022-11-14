#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import re
from argparse import ArgumentParser
from pathlib import Path

from common.VAO import get_date_time, VAO, get_main_transcript_file, get_story_file, get_story_transcripts, \
    read_stories_from_file
from common.constants import TV_FILENAME_RE
from common.fs_utils import frame_idx_to_time, create_dir, add_sec_to_time

parser = ArgumentParser('Automatic Speech Recognition using OpenAI Whisper')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help='Tagesschau video file(s)')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing', help='')


def split_story_transcripts(vao: VAO):
    stories = vao.data.stories
    transcripts = vao.data.transcripts

    for story in stories:
        start = frame_idx_to_time(story.first_frame_idx)
        end = frame_idx_to_time(story.last_frame_idx)

        start = start.replace(microsecond=0)
        end = add_sec_to_time(end, 1)

        yield [transcript for transcript in transcripts if start <= transcript.start and transcript.end <= end]


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
    stories = read_stories_from_file(get_story_file(video))

    return len(story_transcripts) == len(stories)


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    print(f'Split audio of {len(video_files)} videos ... \n')

    for idx, vf in enumerate(video_files):
        vao = VAO(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vao}')

        create_dir(vao.dirs.transcripts_dir, rm_if_exist=True)

        for story_idx, transcript in enumerate(split_story_transcripts(vao)):
            text = ' '.join([transcript.text for transcript in transcript])

            target_file = Path(vao.dirs.transcripts_dir, 'story_' + str(story_idx) + '.txt')

            with open(target_file, 'w') as file:
                file.write(text)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
