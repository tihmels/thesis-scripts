#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import re
from argparse import ArgumentParser
from itertools import tee
from pathlib import Path

import Levenshtein
import numpy as np
import pandas as pd
import spacy
from spellchecker import SpellChecker

from common.VideoData import VideoData, get_shot_file, get_date_time, get_banner_caption_file, get_story_file
from common.constants import TV_FILENAME_RE

parser = ArgumentParser('Scene Segmentation')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')

spell = SpellChecker(language=None, distance=1)  # loads default word frequency list
spell.word_frequency.load_text_file('/Users/tihmels/TS/topics_dict.txt')

spacy_de = spacy.load('de_core_news_sm')


def spellcheck(text):
    print(text)
    return text


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def segment_ts15(vd: VideoData):
    print("WTF")
    yield


def is_named_entity_only(text):
    doc = spacy_de(text)
    entities = doc.ents

    if len(entities) == 1 and len(entities[0].text) == len(text):
        return True


def preprocess_captions(captions):
    for i in range(0, len(captions)):
        _, headline, subline, confidence = captions[i]

        if not (headline.strip() and subline.strip()) or confidence < 0.7:
            predecessor = captions[i - 1]
            captions[i] = predecessor

    return captions


def segment_ts100(vd: VideoData, lev_threshold=5):
    captions = vd.captions[:-1]  # last shot is always weather

    if is_named_entity_only(captions.iloc[0]['headline']):  # check if first banner caption is anchor name
        captions.drop(0, inplace=True)

    captions = preprocess_captions(captions)

    levenshtein_distances = np.empty(len(captions))
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

        story_title = spellcheck(story_title)

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
    df.to_csv(get_story_file(vd))

    return stories


def check_requirements(video: Path):
    if not re.match(TV_FILENAME_RE, video.name):
        return False

    shot_file = get_shot_file(video)

    if not shot_file.is_file():
        print(f'{video.name} has no detected shots.')
        return False

    topic_file = get_banner_caption_file(video)

    if not topic_file.is_file():
        print(f'{video.name} has no topic file.')
        return False

    return True


def was_processed(video: Path):
    return get_story_file(video).is_file()


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    print(f'Scene Segmentation for {len(video_files)} videos ... \n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        segmentor = segment_ts100 if vd.is_summary else segment_ts15
        story_df = segmentor(vd)

        # story_df.to_csv(get_story_file(vd))

        # with open(get_story_file(vd), 'w') as f:
        #     json.dump(story_df, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
