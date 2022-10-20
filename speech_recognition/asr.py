#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import re
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path

import whisper

from common.VideoData import VideoData, get_audio_dir, get_shot_file, get_date_time, \
    get_main_audio_file, get_main_transcript_file
from common.constants import TV_FILENAME_RE

parser = ArgumentParser('Automatic Speech Recognition')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help='Tagesschau video file(s)')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')

model = whisper.load_model("large", in_memory=True)


def transcribe_audio_file(audio_file: Path):
    return model.transcribe(str(audio_file), language='de', fp16=False)


def sec_to_time(seconds):
    return (datetime.min + timedelta(seconds=seconds)).time()


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
    return get_main_transcript_file(video).is_file()


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    print(f'Transcribing audio of {len(video_files)} videos ... \n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        audio_file = get_main_audio_file(vd)

        result = transcribe_audio_file(audio_file)

        transcripts = [(sec_to_time(int(segment['start'])),
                        sec_to_time(int(segment['end'])),
                        segment['text']) for segment in result['segments']]

        with open(get_main_transcript_file(vd), 'w') as file:
            for (start, end, text) in transcripts:
                file.write("[" + start.isoformat() + " - " + end.isoformat() + "]" + " " + text.strip())
                file.write('\n')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
