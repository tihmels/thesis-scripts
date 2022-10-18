#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pydub
from alive_progress import alive_bar

from common.VideoData import VideoData, get_audio_file, get_audio_dir, get_shot_file, get_audio_shot_paths, \
    read_shots_from_file, \
    get_date_time
from common.constants import TV_FILENAME_RE

parser = ArgumentParser()
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing',
                    help="Re-split audio tracks for all videos")

pydub.AudioSegment.ffmpeg = '/usr/local/bin/ffmpeg'


def split_audio_by_scenes(vd: VideoData):
    audio = vd.audio
    scenes = vd.scenes[['first_frame_idx', 'last_frame_idx']].to_records(index=False)

    for scene_idx, (first_frame_idx, last_frame_idx) in enumerate(scenes):
        start_ms = np.divide(first_frame_idx, 25) * 1000
        end_ms = np.divide(last_frame_idx, 25) * 1000

        audio_segment = audio[start_ms:end_ms]

        audio_segment.export(
            Path(vd.audio_dir, 'story_' + str(scene_idx + 1) + '.wav'), format='wav')

        yield


def split_audio_by_shots(vd: VideoData):
    audio = vd.audio
    segments = vd.shots

    for seg_idx, (seg_start_idx, seg_end_idx, _) in enumerate(segments):
        start_ms = np.divide(seg_start_idx, 25) * 1000
        end_ms = np.divide(seg_end_idx, 25) * 1000

        audio_segment = audio[start_ms:end_ms]

        audio_segment.export(
            Path(vd.audio_dir, 'shot_' + str(seg_idx + 1) + '.wav'), format='wav')

        yield


def check_requirements(video: Path):
    if not re.match(TV_FILENAME_RE, video.name):
        return False

    audio_dir = get_audio_dir(video)

    if not audio_dir.is_dir() or get_audio_file(video) is None:
        print(f'{video.name} has no audio file extracted.')
        return False

    shot_file = get_shot_file(video)

    if not shot_file.is_file():
        print(f'{video.name} has no detected shots.')
        return False

    return True


def was_processed(video: Path):
    audio_shots = get_audio_shot_paths(video)

    if len(audio_shots) == len(read_shots_from_file(video)):
        return True

    return False


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0

    video_files = sorted(video_files, key=get_date_time)

    print(f'Splitting audio tracks from {len(video_files)} videos ... \n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        with alive_bar(vd.n_stories, ctrl_c=False, title=f'[{idx + 1}/{len(video_files)}] {vd}', length=20) as bar:
            for _ in split_audio_by_scenes(vd):
                bar()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
