#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import re
from argparse import ArgumentParser
from itertools import tee
from pathlib import Path

import Levenshtein
import numpy as np
import pandas as pd
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

from VideoData import VideoData, get_shot_file, get_date_time, get_caption_file, get_scene_file
from utils.constants import TV_FILENAME_RE


@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    return LanguageDetector()


spacy_de = spacy.load('de_core_news_sm')
spacy_de.add_pipe('language_detector', last=True)


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def segment_ts15(vd: VideoData):
    print("WTF")
    yield


def is_valid_text(text: str):
    doc = spacy_de(text)
    return doc._.language['language'] == 'de' and doc._.language['score'] >= 0.9


def segment_ts100(vd: VideoData, lev_threshold=5):
    captions = {shot_idx: (caption, conf) for shot_idx, caption, conf in vd.captions}

    for i in range(1, len(captions) - 1):
        shot_idx, text, conf = captions[i]

        if not text or conf < 0.5:
            captions[i] = (shot_idx, captions[i - 1][1], '')

    levenshtein_distances = np.empty(vd.n_shots)
    current_caption = ''

    for shot_idx in range(vd.n_shots):
        caption, conf = captions[shot_idx]

        if not caption and conf == 1.0:
            caption = current_caption

        levenshtein_distances[shot_idx] = Levenshtein.distance(current_caption, caption)
        current_caption = caption

    levenshtein_distances = np.where(levenshtein_distances > lev_threshold, 1, 0)

    semantic_boundary_indices = [i for i, x in enumerate(levenshtein_distances) if x == 1]
    semantic_boundary_indices.append(len(captions))

    semantic_boundary_ranges = [(i1, i2 - 1) for i1, i2 in pairwise(semantic_boundary_indices)]

    print(semantic_boundary_ranges)

    stories = []

    for first_shot_idx, last_shot_idx in semantic_boundary_ranges:

        if first_shot_idx == last_shot_idx and captions[first_shot_idx][0] == "":
            continue

        story_captions = [captions.get(idx) for idx in range(first_shot_idx, last_shot_idx + 1)]

        story_title = max(story_captions, key=lambda cap: cap[1])[0]

        story_topic = list(captions.values())[last_shot_idx]
        story_first_shot_idx = int(list(captions.keys())[first_shot_idx])
        story_last_shot_idx = int(list(captions.keys())[last_shot_idx])

        first_frame_idx = vd.shots[story_first_shot_idx][0]
        last_frame_idx = vd.shots[story_last_shot_idx][1]
        from_timestamp = np.divide(first_frame_idx, 25)
        to_timestamp = np.divide(last_frame_idx, 25)

        n_shots = story_last_shot_idx - story_first_shot_idx + 1
        n_frames = last_frame_idx - first_frame_idx + 1
        total_ss = np.round(to_timestamp - from_timestamp, 2)

        data = np.array(
            [story_topic, first_frame_idx, last_frame_idx, n_frames, story_first_shot_idx, story_last_shot_idx, n_shots,
             from_timestamp, to_timestamp, total_ss])

        stories.append(data)
        # stories.append(
        #     {'story_topic': story_topic,
        #      'first_shot_idx': story_first_shot_idx,
        #      'last_shot_idx': story_last_shot_idx,
        #      'n_shots': story_last_shot_idx - story_first_shot_idx + 1,
        #      'first_frame_idx': first_frame_idx,
        #      'last_frame_idx': last_frame_idx,
        #      'n_frames': last_frame_idx - first_frame_idx + 1,
        #      'from_ss': from_timestamp,
        #      'to_ss': to_timestamp,
        #      'total_ss': np.round(to_timestamp - from_timestamp, 2)})

    df = pd.DataFrame(data=stories,
                      columns=['news_title', 'first_frame_idx',
                               'last_frame_idx', 'n_frames', 'first_shot_idx', 'last_shot_idx', 'n_shots', 'from_ss',
                               'to_ss', 'total_ss'])

    df.index = df.index + 1
    df.to_csv(get_scene_file(vd))

    return stories


def check_requirements(path: Path, skip_existing=False):
    match = re.match(TV_FILENAME_RE, path.name)

    if match is None or not path.is_file():
        print(f'{path.name} does not exist or does not match TV-*.mp4 pattern.')
        return False

    shot_file = get_shot_file(path)

    if not shot_file.is_file():
        print(f'{path.name} has no detected shots.')
        return False

    topic_file = get_caption_file(path)

    if not topic_file.is_file():
        print(f'{path.name} has no topic file.')
        return False

    if skip_existing:
        return False

    return True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
    parser.add_argument('-s', '--skip', action='store_true', help="skip scene segmentation if already exist")
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

    print(f'Scene Segmentation for {len(video_files)} videos ... \n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        segmentor = segment_ts100 if vd.is_summary else segment_ts15
        story_df = segmentor(vd)

        # story_df.to_csv(get_story_file(vd))

        # with open(get_story_file(vd), 'w') as f:
        #     json.dump(story_df, f, indent=4, ensure_ascii=False)
