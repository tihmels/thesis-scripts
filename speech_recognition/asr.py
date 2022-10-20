#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import re
from argparse import ArgumentParser
from pathlib import Path

import whisper

from common.VideoData import VideoData, get_audio_dir, get_audio_file, get_shot_file, get_date_time, \
    get_story_audio_files
from common.constants import TV_FILENAME_RE

parser = ArgumentParser()
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')

model = whisper.load_model("large", in_memory=True)


def transcribe_audio(audio_file: Path):
    result = model.transcribe(str(audio_file), language='de', fp16=False)

    return result['text']


def process_video(vd: VideoData):
    trans_dir = vd.transcripts_dir

    for audio_file in get_story_audio_files(vd):
        text = transcribe_audio(audio_file)

        file = Path(trans_dir, audio_file.stem + ".txt")

        with open(file, 'w') as f:
            f.writelines(text.strip())


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
    return False


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    print(f'Transcribing audio streams of {len(video_files)} videos ... \n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        vd.transcripts_dir.mkdir(exist_ok=True)

        process_video(vd)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
