#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import re
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import spacy
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from common.DataModel import TranscriptData
from common.Schemas import STORY_COLUMNS
from common.VAO import VAO, get_shot_file, get_date_time, get_banner_caption_file, get_story_file, \
    get_topic_file, is_summary, read_shots_from_file
from common.constants import TV_FILENAME_RE
from common.fs_utils import frame_idx_to_sec, sec_to_time

parser = ArgumentParser('Story Segmentation')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')

spacy_de = spacy.load('de_core_news_sm')
spacy_de_lg = spacy.load('de_core_news_lg')

# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
# model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

nltk.download('stopwords')
german_stop_words = stopwords.words('german')


def get_text(tds: [TranscriptData]):
    return ' '.join([td.text for td in tds])


def to_datetime(time):
    return datetime(2000, 1, 1, time.hour, time.minute, time.second)


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


def get_count_vectorizer(text) -> CountVectorizer:
    # vectorizer = CountVectorizer(lowercase=True, stop_words=german_stop_words, preprocessor=spacy_lemmatizer)
    vectorizer = CountVectorizer(lowercase=True, stop_words=german_stop_words)
    vectorizer.fit([text])

    return vectorizer


def spacy_lemmatizer(text):
    lemmatized = [token.lemma_ for token in spacy_de_lg(text)]
    return ' '.join(lemmatized)


def topic_to_anchor_by_transcript(topics, anchor_shots, anchor_transcripts):
    dt_matrix = np.zeros(shape=(len(topics), len(anchor_shots)), dtype=np.int8)

    for idx, topic in topics.items():
        vectorizer = get_count_vectorizer(topic)
        bow = vectorizer.transform([get_text(tds) for tds in anchor_transcripts.values()])

        bow_sum = [sum(vec) for vec in bow.toarray()]
        dt_matrix[idx] = bow_sum

    argmax, values = np.argmax(dt_matrix, axis=1), np.max(dt_matrix, axis=1)

    to_shot_idx = np.vectorize(lambda dt_idx: list(anchor_shots.keys())[dt_idx])
    shot_idxs = to_shot_idx(argmax)

    return {topic_idx: shot_idx for idx, (topic_idx, shot_idx) in enumerate(zip(topics, shot_idxs)) if
            (values[idx] > 0 and all(pre_shot_idx < shot_idx for pre_shot_idx in shot_idxs[:idx]))}


def get_anchor_transcripts(vao: VAO, anchor_shots, max_shots=5):
    return {
        idx: vao.get_shot_transcripts(idx,
                                      min(next(
                                          (next_idx - 1 for next_idx in anchor_shots.keys() if next_idx > idx),
                                          idx), idx + max_shots, vao.n_shots))
        for idx in anchor_shots}


def extract_story_data(vao: VAO, first_shot_idx: int, last_shot_idx: int):
    n_shots = last_shot_idx - first_shot_idx + 1

    first_frame_idx = vao.data.shots[first_shot_idx].first_frame_idx
    last_frame_idx = vao.data.shots[last_shot_idx].last_frame_idx

    n_frames = last_frame_idx - first_frame_idx + 1

    from_time = sec_to_time(frame_idx_to_sec(first_frame_idx))
    to_time = sec_to_time(frame_idx_to_sec(last_frame_idx))

    timedelta = to_datetime(to_time) - to_datetime(from_time)
    total_ss = timedelta.total_seconds()

    return (first_frame_idx, last_frame_idx, n_frames, first_shot_idx, last_shot_idx, n_shots,
            from_time.strftime('%H:%M:%S'), to_time.strftime('%H:%M:%S'), total_ss)


def segment_ts15(vao: VAO):
    anchor_shots = {idx: shot for idx, shot in enumerate(vao.data.shots) if shot.type == 'anchor'}
    news_topics = {idx: title for idx, title in enumerate(vao.data.topics[:-1]) if
                   not 'Lottozahlen' in title}  # last shot is always weather

    shot_cutoff_idx = next(
        idx for idx, shot in enumerate(vao.data.shots) if shot.type == 'weather' or shot.type == 'lotto')
    anchor_shots = {idx: sd for idx, sd in anchor_shots.items() if idx < shot_cutoff_idx}

    anchor_transcripts = get_anchor_transcripts(vao, anchor_shots, 3)

    topic_to_anchor = topic_to_anchor_by_transcript(news_topics, anchor_shots, anchor_transcripts)

    missing_topics = [idx for idx in news_topics.keys() if idx not in topic_to_anchor.keys()]
    if len(missing_topics) > 0:
        print(f'{missing_topics} could not be assigned!')
    else:
        print()

    stories = []

    for topic_idx, anchor_idx in topic_to_anchor.items():
        story_title = news_topics[topic_idx]

        first_shot_idx = anchor_idx
        last_shot_idx = next((next_idx for next_idx in list(topic_to_anchor.values()) if next_idx > anchor_idx),
                             list(anchor_shots.keys())[-1]) - 1

        story_data = extract_story_data(vao, first_shot_idx, last_shot_idx)

        stories.append((story_title, *story_data))

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


def segment_ts100(vao: VAO):
    captions = {idx: cd for idx, cd in enumerate(vao.data.captions[:-1])}  # last shot is always weather

    preprocess_captions(captions)

    story_indices = get_story_indices_by_captions(captions)

    stories = []

    for story in story_indices:

        story_captions = [captions[idx] for idx in story]

        if story[0] == story[-1] and story_captions[0].text == "":
            continue

        story_title = max(story_captions, key=lambda k: k.confidence).text

        story_data = extract_story_data(vao, min(story), max(story))

        stories.append((story_title, *story_data))

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


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    print(f'Story Segmentation for {len(video_files)} videos ... \n')

    for idx, vf in enumerate(video_files):
        vao = VAO(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vao} ', end='... ')

        segmentor = segment_ts100 if vao.is_summary else segment_ts15

        df = segmentor(vao)
        df.to_csv(get_story_file(vao), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
