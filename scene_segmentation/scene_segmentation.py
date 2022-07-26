#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import json
import re
from argparse import ArgumentParser
from itertools import tee
from pathlib import Path

import Levenshtein
import numpy as np
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

from VideoData import VideoData, get_shot_file, get_date_time, get_topic_file, get_story_file
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
    topics = {shot_idx: topic for shot_idx, topic in vd.topics if topic and is_valid_text(topic)}
    print(topics)

    levenshtein_distances = np.empty(len(topics))
    current_topic = ''

    for index, topic in enumerate(topics.values()):
        levenshtein_distances[index] = Levenshtein.distance(current_topic, topic)
        current_topic = topic

    print(levenshtein_distances)

    levenshtein_distances = np.where(levenshtein_distances > lev_threshold, 1, 0)
    print(levenshtein_distances)

    semantic_boundary_indices = [i for i, x in enumerate(levenshtein_distances) if x == 1]
    semantic_boundary_indices.append(len(topics))
    print(semantic_boundary_indices)

    semantic_boundary_ranges = [(i1, i2 - 1) for i1, i2 in pairwise(semantic_boundary_indices)]
    print(semantic_boundary_ranges)

    stories = []

    for first_shot_idx, last_shot_idx in semantic_boundary_ranges:
        story_topic = list(topics.values())[last_shot_idx]
        story_first_shot_idx = int(list(topics.keys())[first_shot_idx])
        story_last_shot_idx = int(list(topics.keys())[last_shot_idx])

        stories.append(
            {'topic': story_topic,
             'first_shot_idx': story_first_shot_idx,
             'last_shot_idx': story_last_shot_idx,
             'first_frame_idx': vd.shots[story_first_shot_idx][0],
             'last_frame_idx': vd.shots[story_last_shot_idx][1]})


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

    topic_file = get_topic_file(path)

    if not topic_file.is_file():
        print(f'{path.name} has no topic file.')
        return False

    if skip_existing:
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

    print(f'Scene Segmentation for {len(video_files)} videos ... \n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        segmentor = segment_ts100 if vd.is_summary else segment_ts15
        stories = segmentor(vd)

        with open(get_story_file(vd), 'w') as f:
            json.dump(stories, f, indent=4, ensure_ascii=False)
