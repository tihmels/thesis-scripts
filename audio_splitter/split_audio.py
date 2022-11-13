#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pydub
from alive_progress import alive_bar

from common.VAO import get_main_audio_file, get_audio_dir, get_shot_file, get_shot_audio_files, read_shots_from_file, \
    get_date_time, VAO
from common.constants import TV_FILENAME_RE

pydub.AudioSegment.ffmpeg = '/usr/local/bin/ffmpeg'

parser = ArgumentParser()
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help='Tagesschau video file(s)')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing',
                    help="Re-split audio tracks for all videos")


def split_audio_by_scenes(vao: VAO):
    audio = get_main_audio_file(vao)
    stories = vao.data.stories

    for story_idx, story in enumerate(stories):
        start_ms = np.divide(story.first_frame_idx, 25) * 1000
        end_ms = np.divide(story.last_frame_idx, 25) * 1000

        audio_segment = audio[start_ms:end_ms]

        audio_segment.export(
            Path(vao.dirs.audio_dir, 'story_' + str(story_idx + 1) + '.wav'), format='wav')

        yield


def split_audio_by_shots(vao: VAO):
    audio = vao.data.audio
    shots = vao.data.shots

    for shot_idx, sd in enumerate(shots):
        start_ms = np.divide(sd.first_frame_idx, 25) * 1000
        end_ms = np.divide(sd.last_frame_idx, 25) * 1000

        audio_segment = audio[start_ms:end_ms]

        audio_segment.export(
            Path(vao.dirs, 'shot_' + str(shot_idx + 1) + '.wav'), format='wav')

        yield


def check_requirements(video: Path):
    if not re.match(TV_FILENAME_RE, video.name):
        return False

    audio_dir = get_audio_dir(video)

    if not audio_dir.is_dir() or get_main_audio_file(video) is None:
        print(f'{video.name} has no audio file extracted.')
        return False

    shot_file = get_shot_file(video)

    if not shot_file.is_file():
        print(f'{video.name} has no detected shots.')
        return False

    return True


def was_processed(video: Path):
    audio_shots = get_shot_audio_files(video)

    return len(audio_shots) == len(read_shots_from_file(video))


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0

    video_files = sorted(video_files, key=get_date_time)

    print(f'Splitting audio tracks from {len(video_files)} videos ... \n')

    for idx, vf in enumerate(video_files):
        vao = VAO(vf)

        with alive_bar(vao.n_stories, ctrl_c=False, title=f'[{idx + 1}/{len(video_files)}] {vao}', length=20) as bar:
            for _ in split_audio_by_shots(vao):
                bar()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
