#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import re
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pandas as pd
import spacy
import yake
from fuzzywuzzy import fuzz
from transformers import T5ForConditionalGeneration, T5Tokenizer

from common.DataModel import TranscriptData
from common.Schemas import STORY_COLUMNS
from common.VideoData import VideoData, get_shot_file, get_date_time, get_banner_caption_file, get_story_file, \
    get_topic_file, is_summary, read_shots_from_file
from common.constants import TV_FILENAME_RE
from common.fs_utils import frame_idx_to_sec, sec_to_time

parser = ArgumentParser('Story Segmentation')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')

spacy_de = spacy.load('de_core_news_sm')
simple_kw_extractor = yake.KeywordExtractor(lan='de', top=1, n=2)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")


def get_text(tds: [TranscriptData]):
    return ' '.join([td.text for td in tds])


def is_matching_headline(headline, text):
    input_text = f'Headline: "{headline}". Story: "{text}". Q: Does the Headline fit the Story?'

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    output = tokenizer.decode(outputs[0])

    return 'yes' in output.lower()


def anchor_topic_detection(topics, anchor_shots, anchor_transcripts):
    anchor_to_topic = {}

    for topic_idx, headline in topics.items():

        anchorshot_idxs = [idx for idx in anchor_shots.keys() if idx not in anchor_to_topic.keys()]

        for anchor_idx in anchorshot_idxs:

            transcript = get_text(anchor_transcripts[anchor_idx])

            if is_matching_headline(headline, transcript):
                anchor_to_topic[anchor_idx] = topic_idx
                break

    return anchor_to_topic


def segment_ts15(vd: VideoData):
    shots = {idx: shot for idx, shot in enumerate(vd.shots)}
    topics = {idx: topic for idx, topic in enumerate(vd.topics)}

    first_weather_shot = next(idx for idx, shot in shots.items() if shot.type == 'weather')

    anchor_shots = {idx: sd for idx, sd in shots.items() if sd.type == 'anchor'}

    anchor_shots_trimmed = {idx: sd for idx, sd in anchor_shots.items() if idx < first_weather_shot}
    anchor_transcripts = {
        idx: vd.get_shot_transcripts(idx,
                                     min(next((next_idx - 1 for next_idx in anchor_shots_trimmed.keys() if next_idx > idx),
                                              idx), idx + 5, vd.n_shots))
        for idx in anchor_shots}

    anchor_to_topic = anchor_topic_detection(topics, anchor_shots, anchor_transcripts)
    anchor_keys = list(sorted(anchor_to_topic.keys()))

    missing_topics = [idx for idx in topics.keys() if idx not in anchor_to_topic.values()]
    if len(missing_topics) > 0:
        print(f'Topics {missing_topics} could not be assigned!')

    story_indices = [list(range(a1, next((next_idx for next_idx in anchor_shots.keys() if next_idx > a1)))) for a1
                     in anchor_keys]

    # story_indices = [list(range(a1, a2)) for a1, a2 in pairwise(anchor_keys)]

    stories = []

    for story in story_indices[:-1]:
        story_title = topics[anchor_to_topic[story[0]]]

        first_shot_idx, last_shot_idx = min(story), max(story)
        first_frame_idx, last_frame_idx = vd.shots[first_shot_idx].first_frame_idx, vd.shots[
            last_shot_idx].last_frame_idx

        from_timestamp = sec_to_time(frame_idx_to_sec(first_frame_idx))
        to_timestamp = sec_to_time(frame_idx_to_sec(last_frame_idx))

        n_shots = last_shot_idx - first_shot_idx + 1
        n_frames = last_frame_idx - first_frame_idx + 1

        timedelta = to_datetime(to_timestamp) - to_datetime(from_timestamp)
        total_ss = timedelta.total_seconds()

        data = (story_title,
                first_frame_idx, last_frame_idx, n_frames,
                first_shot_idx, last_shot_idx, n_shots,
                from_timestamp.strftime('%H:%M:%S'), to_timestamp.strftime('%H:%M:%S'), total_ss)

        stories.append(data)

    df = pd.DataFrame(data=stories, columns=STORY_COLUMNS)

    return df


def is_named_entity_only(text):
    doc = spacy_de(text)
    entities = doc.ents

    return len(entities) == 1 and len(entities[0].text) == len(text)


def preprocess_captions(captions):
    if is_named_entity_only(captions[0].text):  # check if first banner text is anchor name
        captions.pop(0)

    # the last shot in a news story is often very short and the banner disappears early,
    # which is why the banner text is not captured by OCR.
    # we fix this by assuming that this shot belongs to the previous caption.
    for idx, cd in captions.items():
        if (not cd.text.strip() or cd.confidence < 0.7) and idx - 1 in captions:
            predecessor = captions[idx - 1]
            captions[idx] = predecessor


def get_story_indices_by_captions(captions):
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

    return story_indices


def to_datetime(time):
    return datetime(2000, 1, 1, time.hour, time.minute, time.second)


def segment_ts100(vd: VideoData):
    captions = {idx: cd for idx, cd in enumerate(vd.captions[:-1])}  # last shot is always weather

    preprocess_captions(captions)

    story_indices = get_story_indices_by_captions(captions)

    stories = []

    for story in story_indices:

        story_captions = [captions[idx] for idx in story]

        if story[0] == story[-1] and story_captions[0].text == "":
            continue

        story_title = max(story_captions, key=lambda k: k.confidence).text

        first_shot_idx, last_shot_idx = min(story), max(story)
        first_frame_idx, last_frame_idx = vd.shots[first_shot_idx].first_frame_idx, vd.shots[
            last_shot_idx].last_frame_idx

        from_timestamp = sec_to_time(frame_idx_to_sec(first_frame_idx))
        to_timestamp = sec_to_time(frame_idx_to_sec(last_frame_idx))

        n_shots = last_shot_idx - first_shot_idx + 1
        n_frames = last_frame_idx - first_frame_idx + 1

        timedelta = to_datetime(to_timestamp) - to_datetime(from_timestamp)
        total_ss = timedelta.total_seconds()

        data = (story_title,
                first_frame_idx, last_frame_idx, n_frames,
                first_shot_idx, last_shot_idx, n_shots,
                from_timestamp.strftime('%H:%M:%S'), to_timestamp.strftime('%H:%M:%S'), total_ss)

        stories.append(data)

    df = pd.DataFrame(data=stories, columns=STORY_COLUMNS)

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


def extract_keywords(text):
    return simple_kw_extractor.extract_keywords(text)[0]


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    print(f'Story Segmentation for {len(video_files)} videos ... \n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vd} ', end='... ')

        segmentor = segment_ts100 if vd.is_summary else segment_ts15

        df = segmentor(vd)
        df.to_csv(get_story_file(vd), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
