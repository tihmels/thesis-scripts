#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from pydub import AudioSegment

from VideoData import VideoData
from utils.constants import TV_FILENAME_RE
from utils.fs_utils import get_date_time, get_audio_dir, print_progress_bar, read_segments_from_file, get_shot_file, \
    get_audio_file, get_audio_shots


def process_video(vd: VideoData):
    audio = AudioSegment.from_wav(get_audio_file(vd.path))

    segments = vd.segments

    for seg_idx, (seg_start_idx, seg_end_idx) in enumerate(segments):
        start_ms = np.divide(seg_start_idx, 25) * 1000
        end_ms = np.divide(seg_end_idx, 25) * 1000

        audio_segment = audio[start_ms:end_ms]
        audio_segment.export(Path(vd.audio_dir, 'shot_' + str(seg_idx + 1) + '.wav', format='wav'))

        print_progress_bar(seg_idx + 1, len(vd.segments), length=25)


def check_requirements(path: Path, skip_existing=False):
    match = re.match(TV_FILENAME_RE, path.name)

    if match is None or not path.is_file():
        print(f'{path.name} does not exist or does not match TV-*.mp4 pattern.')
        return False

    audio_dir = get_audio_dir(path)

    if not audio_dir.is_dir() or get_audio_file(path) is None:
        print(f'{path.name} has no audio file extracted.')
        return False

    if skip_existing:
        audio_shots = get_audio_shots(path)

        if len(audio_shots) == len(read_segments_from_file(get_shot_file(path))):
            return False

    return True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
    parser.add_argument('-s', '--skip', action='store_true', help="skip keyframe extraction if already exist")
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    video_files = []

    for file in args.files:
        if file.is_file() and check_requirements(file, args.skip):
            video_files.append(file)
        elif file.is_dir():
            video_files.extend([video for video in file.glob('*.mp4') if check_requirements(video, args.skip)])

    assert len(video_files) > 0

    video_files.sort(key=get_date_time)

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        print(f'\n[{idx + 1}/{len(video_files)}] {vd}')
        process_video(vd)
