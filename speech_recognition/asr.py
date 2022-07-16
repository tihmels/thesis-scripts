#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import json
import os
import re
from argparse import ArgumentParser
from pathlib import Path

from VideoData import get_date_time, VideoData, get_audio_dir, get_audio_file, get_shot_file, get_transcript_file
from utils.constants import TV_FILENAME_RE

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from huggingsound import SpeechRecognitionModel, KenshoLMDecoder

lm_path = "/Users/tihmels/Scripts/thesis-scripts/speech_recognition/model/lm.binary"
unigrams_path = "/Users/tihmels/Scripts/thesis-scripts/speech_recognition/model/unigrams.txt"

JG_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-german"
FB_MODEL_ID = "facebook/wav2vec2-large-xlsr-53-german"
model = SpeechRecognitionModel(JG_MODEL_ID)

decoder = KenshoLMDecoder(model.token_set, lm_path=lm_path, unigrams_path=unigrams_path)


def transcribe_audio(vd: VideoData):
    audio_file = get_audio_file(vd)

    return model.transcribe([audio_file])[0]


def check_requirements(path: Path, skip_existing=False):
    match = re.match(TV_FILENAME_RE, path.name)

    if match is None or not path.is_file():
        print(f'{path.name} does not exist or does not match TV-*.mp4 pattern.')
        return False

    audio_dir = get_audio_dir(path)

    if not audio_dir.is_dir() or get_audio_file(path) is None:
        print(f'{path.name} has no audio file extracted.')
        return False

    shot_file = get_shot_file(path)

    if not shot_file.is_file():
        print(f'{path.name} has no detected shots.')
        return False

    if skip_existing:
        transcript_file = get_transcript_file(path)

        if transcript_file.is_file():
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

    print(f'Transcribing audio stream of {len(video_files)} videos ... \n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        result = transcribe_audio(vd)

        with open(get_transcript_file(vd), 'w') as file:
            json.dump(result, file, ensure_ascii=False)
