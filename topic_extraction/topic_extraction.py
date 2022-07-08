#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import json
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import skimage
import io
import codecs
from PIL import Image
from alive_progress import alive_bar
from pytesseract import pytesseract, Output
from skimage.filters.edges import sobel
from skimage.filters.thresholding import try_all_threshold

from VideoData import get_date_time, VideoData, get_kf_dir, get_keyframe_paths, get_topic_file
from utils.constants import TV_FILENAME_RE


def extract_topics(vd):
    keyframes = [Image.open(frame).convert('L') for frame in vd.kfs]
    keyframes = [np.array(frame) for frame in keyframes]

    for frame in keyframes:
        caption_box = frame[-48:-18, -425:]
        thresh = skimage.filters.thresholding.threshold_li(caption_box)
        caption_box = caption_box > thresh

        caption_text = pytesseract.image_to_string(caption_box, output_type=Output.DICT, lang='deu',
                                                   config='--psm 7 --oem 1')

        # Image.fromarray(caption_box).convert('RGB').save("/Users/tihmels/Desktop/cb.jpg")
        yield caption_text['text']


def check_requirements(path: Path, skip_existing=False):
    assert path.parent.name == 'ts100'

    match = re.match(TV_FILENAME_RE, path.name)

    if match is None or not path.is_file():
        print(f'{path.name} does not exist or does not match TV-*.mp4 pattern.')
        return False

    keyframe_dir = get_kf_dir(path)

    if not keyframe_dir.exists() or len(get_keyframe_paths(path)) < 1:
        print(f'{path.name} has no keyframes extracted.')
        return False

    return True
    pass


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

    print(f'Splitting audio shots from {len(video_files)} videos ... \n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        with alive_bar(vd.n_shots, ctrl_c=False, title=f'[{idx + 1}/{len(video_files)}] {vd}', length=20) as bar:

            topics = {}

            escapes = ''.join([chr(char) for char in range(1, 32)])
            translator = str.maketrans('', '', escapes)

            for shot_idx, text in enumerate(extract_topics(vd)):
                unescaped_text = text.translate(translator)
                topics[shot_idx] = unescaped_text

                bar()

            json_object = json.dumps(topics, indent=4, ensure_ascii=False)
            print(json_object)

            with open(get_topic_file(vd), 'w') as f:
                json.dump(topics, f, ensure_ascii=False)
