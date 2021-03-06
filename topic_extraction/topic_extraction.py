#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import json
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import skimage
import spacy
from PIL import Image
from alive_progress import alive_bar
from pytesseract import pytesseract, Output
from skimage.filters.edges import sobel
from skimage.filters.thresholding import try_all_threshold
from spellchecker import SpellChecker

from VideoData import get_date_time, VideoData, get_kf_dir, get_keyframe_paths, get_topic_file
from utils.constants import TV_FILENAME_RE

spell = SpellChecker(language='de')
spell.word_frequency.load_text_file('/Users/tihmels/TS/topics_dict.txt')
spell.word_frequency.load_words(
    ['Windräder', 'Corona-Expertinnenrat', 'Hürden', 'Ausweitung', 'Fußball-Bundesliga', 'Fallzahlen',
     'Vorbereitungen'])

spacy_de = spacy.load('de_core_news_sm')


def spellcheck(word: str):
    misspelled = spell.unknown(list(word))

    for w in misspelled:
        print(w)
        return spell.correction(w)

    return word


def extract_topics(vd, resize_factor=2):
    first_shot_indices = [idx1 for idx1, idx2 in vd.shots]

    keyframes = [Image.open(frame).convert('L') for frame in np.array(vd.frames)[first_shot_indices]]
    keyframes = [frame.resize((frame.size[0] * resize_factor, frame.size[1] * resize_factor)) for frame in keyframes]
    keyframes = [np.array(frame) for frame in keyframes]

    for frame in keyframes:
        caption_box = frame[-48 * resize_factor:-18 * resize_factor, -425 * resize_factor:]
        thresh = skimage.filters.thresholding.threshold_li(caption_box)
        caption_box = caption_box > thresh

        Image.fromarray(caption_box).save("/Users/tihmels/Desktop/test.png")

        custom_oem_psm_config = r'--psm 4 --oem 1'
        caption_data = pytesseract.image_to_data(caption_box, output_type=Output.DICT, lang='deu+eng',
                                                 config=custom_oem_psm_config)

        yield caption_data


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

    if skip_existing and get_topic_file(path).exists():
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

    print(f'Extract topics from {len(video_files)} videos ... \n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        with alive_bar(vd.n_shots, ctrl_c=False, title=f'[{idx + 1}/{len(video_files)}] {vd}', length=20) as bar:

            topics = {}

            for shot_idx, text_data in enumerate(extract_topics(vd)):

                words = np.array([word for word in text_data['text'] if word])
                confidence_levels = np.array([conf for conf in text_data['conf'] if conf > 0])

                # low_confidence_indices = np.argwhere(confidence_levels < 50)

                # for lc_idx in low_confidence_indices:
                #    words[lc_idx] = spellcheck(words[lc_idx])

                text = ' '.join(words).strip()

                doc = spacy_de(text)
                entities = doc.ents

                if len(entities) == 1 and len(entities[0]) == len(doc):
                    text = ""

                topics[shot_idx] = text

                bar()

            json_object = json.dumps(topics, indent=4, ensure_ascii=False)
            print(json_object)

            with open(get_topic_file(vd), 'w') as f:
                json.dump(topics, f, ensure_ascii=False)
