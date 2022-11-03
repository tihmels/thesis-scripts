#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import yake
from fuzzywuzzy import fuzz
from spellchecker import SpellChecker
from transformers import T5Tokenizer, T5ForConditionalGeneration

from common.VideoData import VideoData, get_shot_file, get_date_time, get_banner_caption_file, get_story_file, \
    get_topic_file, is_summary, read_shots_from_file
from common.constants import TV_FILENAME_RE

parser = ArgumentParser('Scene Segmentation')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')

spell = SpellChecker(language=None, distance=1)  # loads default word frequency list
spell.word_frequency.load_text_file('/Users/tihmels/TS/topics_dict.txt')

spacy_de = spacy.load('de_core_news_sm')
simple_kw_extractor = yake.KeywordExtractor(lan='de', top=1, n=2)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))


def segment_ts15(vd: VideoData):
    shots = {idx: shot for idx, shot in enumerate(vd.shots)}
    topics = vd.topics

    is_anchorshot = lambda c: c.clazz == 'anchor'

    anchorshot_indices = [idx for idx, clazz in classifications.items() if is_anchorshot(clazz)]

    anchor_transcripts = [vd.get_shot_transcripts(idx) for idx in anchorshot_indices]

    topic = vd.topics[0]
    transcript = ' '.join(anchor_transcripts[0])

    return None


def is_named_entity_only(text):
    doc = spacy_de(text)
    entities = doc.ents

    return len(entities) == 1 and len(entities[0].text) == len(text)


def preprocess_captions(captions):
    if is_named_entity_only(captions[0].text):  # check if first banner text contains anchor name
        captions.pop(0)
    # the last shot in a story is often very short, which is why the banner text is not captured by OCR.
    # we fix this assuming that this shot belongs to the previous caption
    for idx, cd in captions.items():
        if (not cd.text.strip() or cd.confidence < 0.7) and idx - 1 in captions:
            predecessor = captions[idx - 1]
            captions[idx] = predecessor


def segment_ts100(vd: VideoData):
    captions = {idx: cd for idx, cd in enumerate(vd.captions[:-1])}  # last shot is always weather

    preprocess_captions(captions)

    first_idx, last_idx = min(captions.keys()), max(captions.keys())
    current_caption = captions[first_idx].text

    story_indices = [[first_idx]]

    for idx in range(first_idx + 1, last_idx):
        next_caption = captions[idx].text

        ratio = fuzz.token_sort_ratio(current_caption, next_caption)

        if ratio > 80:
            story_indices[-1].append(idx)
        else:
            story_indices.append([idx])

        current_caption = next_caption

    stories = []

    for story in story_indices:

        story_captions = [captions[idx] for idx in story]

        if story[0] == story[-1] and story_captions[0].text == "":
            continue

        story_title = max(story_captions, key=lambda k: k.confidence).text

        first_shot_idx, last_shot_idx = min(story), max(story)
        first_frame_idx, last_frame_idx = vd.shots[first_shot_idx][0], vd.shots[last_shot_idx][1]

        from_timestamp = np.divide(first_frame_idx, 25)
        to_timestamp = np.divide(last_frame_idx, 25)

        n_shots = last_shot_idx - first_shot_idx + 1
        n_frames = last_frame_idx - first_frame_idx + 1
        total_ss = np.round(to_timestamp - from_timestamp, 2)

        data = (story_title,
                first_frame_idx, last_frame_idx, n_frames,
                first_shot_idx, last_shot_idx, n_shots,
                from_timestamp, to_timestamp, total_ss)

        stories.append(data)

    df = pd.DataFrame(data=stories,
                      columns=['news_title', 'first_frame_idx',
                               'last_frame_idx', 'n_frames', 'first_shot_idx', 'last_shot_idx', 'n_shots', 'from_ss',
                               'to_ss', 'total_ss'])

    return df


def check_requirements(video: Path):
    if not re.match(TV_FILENAME_RE, video.name):
        return False

    shot_file = get_shot_file(video)

    if not shot_file.is_file():
        print(f'{video.name} has no detected shots.')
        return False

    if not is_summary(video) and any(shot.type is None for shot in read_shots_from_file(shot_file)):
        print(f'{video.name} has no detected shot types.')
        return False

    topic_file = get_banner_caption_file(video) if is_summary(video) else get_topic_file(video)

    if not topic_file.is_file():
        print(f'{video.name} has no topic file.')
        return False

    return True


def was_processed(video: Path):
    return get_story_file(video).is_file()


def extract_keyword(text):
    language = "de"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    n_keywords = 1
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                                top=n_keywords, features=None)

    return simple_kw_extractor.extract_keywords(text)[0]


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    print(f'Scene Segmentation for {len(video_files)} videos ... \n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vd} | ', end='')

        segmentor = segment_ts100 if vd.is_summary else segment_ts15
        df = segmentor(vd)

        df.to_csv(get_story_file(vd), index=False)

        news_stories = df['news_title'].values.tolist()
        keywords = [extract_keyword(text) for text in news_stories]

        print(' '.join([str(sidx + 1) + f'. {kw[0]}' for sidx, kw in enumerate(keywords)]))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)