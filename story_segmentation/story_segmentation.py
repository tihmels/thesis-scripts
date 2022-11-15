#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import operator
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from HanTa import HanoverTagger as ht
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from charsplit.splitter import Splitter
from common.DataModel import TranscriptData
from common.Schemas import STORY_COLUMNS
from common.VAO import VAO, get_shot_file, get_date_time, get_banner_caption_file, get_story_file, \
    get_topic_file, is_summary, read_shots_from_file
from common.constants import TV_FILENAME_RE
from common.fs_utils import frame_idx_to_sec, sec_to_time, time_to_datetime

parser = ArgumentParser('Story Segmentation')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')

spacy_de = spacy.load('de_core_news_sm')
german_stop_words = stopwords.words('german')

tagger = ht.HanoverTagger('morphmodel_ger.pgz')
splitter = Splitter()


def lemmatizer(text):
    return [tag for word in text for tag in tagger.analyze(word)[:1]]


def get_text(tds: [TranscriptData]):
    return ' '.join([td.text for td in tds])


def extract_story_data(vao: VAO, first_shot_idx: int, last_shot_idx: int):
    n_shots = last_shot_idx - first_shot_idx + 1

    first_frame_idx = vao.data.shots[first_shot_idx].first_frame_idx
    last_frame_idx = vao.data.shots[last_shot_idx].last_frame_idx

    n_frames = last_frame_idx - first_frame_idx + 1

    from_time = sec_to_time(frame_idx_to_sec(first_frame_idx))
    to_time = sec_to_time(frame_idx_to_sec(last_frame_idx))

    timedelta = time_to_datetime(to_time) - time_to_datetime(from_time)
    total_ss = timedelta.total_seconds()

    return (first_frame_idx, last_frame_idx, n_frames, first_shot_idx, last_shot_idx, n_shots,
            from_time.strftime('%H:%M:%S'), to_time.strftime('%H:%M:%S'), total_ss)


def umlauts(text):
    """
    Replace umlauts for a given text

    :param text: text as string
    :return: manipulated text as str
    """

    temp_var = text  # local variable

    temp_var = temp_var.replace('ä', 'ae')
    temp_var = temp_var.replace('ö', 'oe')
    temp_var = temp_var.replace('ü', 'ue')
    temp_var = temp_var.replace('Ä', 'Ae')
    temp_var = temp_var.replace('Ö', 'Oe')
    temp_var = temp_var.replace('Ü', 'Ue')
    temp_var = temp_var.replace('ß', 'ss')

    return temp_var


abbreviations = {'AKW': 'Atomkraftwerk', 'WM': 'Weltmeisterschaft'}


def abbrvs(text):
    for abbrv, sub in abbreviations.items():
        text = re.sub(abbrv, sub, text)
    return text


def custom_preprocessor(text):
    text = abbrvs(text)

    text = text.lower()
    text = umlauts(text)

    text = re.sub("\\W", " ", text)  # remove special chars

    german_stop_words_to_use = []  # List to hold words after conversion

    for word in german_stop_words:
        german_stop_words_to_use.append(umlauts(word))

    text_wo_stop_words = [word for word in text.split() if word.lower() not in german_stop_words_to_use]

    return text_wo_stop_words


def custom_tokenizer(text):
    splits = [list(split[1:]) for word in text for split in splitter.split_compound(word) if split[0] > 0.9]
    splits = [item for t in splits for item in t]

    return lemmatizer(text + splits)


def get_vectorizer(text) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(preprocessor=custom_preprocessor, tokenizer=custom_tokenizer, token_pattern=None)
    vectorizer.fit([text])

    return vectorizer


def topic_to_anchor_by_transcript(topics, anchor_shots, anchor_transcripts):
    dt_matrix = np.zeros(shape=(len(topics), len(anchor_shots)))

    for idx, topic in topics.items():
        vectorizer = get_vectorizer(topic)
        bow = vectorizer.transform([transcript for transcript in anchor_transcripts.values()])

        voc = vectorizer.vocabulary_

        bow_sum = [sum(vec) for vec in bow.toarray()]
        dt_matrix[idx] = bow_sum

    argmax, values = np.argmax(dt_matrix, axis=1), np.max(dt_matrix, axis=1)

    to_shot_idx = np.vectorize(lambda dt_idx: list(anchor_shots.keys())[dt_idx])
    shot_idxs = to_shot_idx(argmax)

    return {topic_idx: shot_idx for idx, (topic_idx, shot_idx) in enumerate(zip(topics, shot_idxs)) if
            (values[idx] > 0 and all(pre_shot_idx < shot_idx for pre_shot_idx in shot_idxs[:idx]))}


def get_anchor_transcripts(vao: VAO, anchor_shots, max_shots=5):
    return {
        idx: get_text(vao.get_shot_transcripts(idx,
                                               min(next(
                                                   (next_idx - 1 for next_idx in anchor_shots.keys() if next_idx > idx),
                                                   idx), idx + max_shots, vao.n_shots)))
        for idx in anchor_shots}


def post_processing(topic_to_anchor, news_topics, anchor_shots):
    assigned_topics = topic_to_anchor.keys()
    missing_topics = [idx for idx in news_topics.keys() if idx not in assigned_topics]

    for topic_idx in missing_topics:
        if topic_idx - 1 in assigned_topics and topic_idx + 1 in assigned_topics and len(
                [idx for idx in anchor_shots.keys() if
                 topic_to_anchor[topic_idx - 1] < idx < topic_to_anchor[topic_idx + 1]]) == 1:
            topic_to_anchor[topic_idx] = next(
                idx for idx in anchor_shots.keys() if idx > topic_to_anchor[topic_idx - 1])

    return {topic_idx: shot_idx for topic_idx, shot_idx in
            sorted(topic_to_anchor.items(), key=operator.itemgetter(1))}


def segment_ts15(vao: VAO):
    topics_cutoff_idx = next(
        idx for idx, topic in enumerate(vao.data.topics) if re.search(r"^(Die Lottozahlen|Das Wetter)$", topic))
    news_topics = {idx: title for idx, title, in enumerate(vao.data.topics) if idx < topics_cutoff_idx}

    shot_cutoff_idx = next(
        idx for idx, shot in enumerate(vao.data.shots) if shot.type == 'weather' or shot.type == 'lotto')
    anchor_shots = {idx: sd for idx, sd in enumerate(vao.data.shots) if sd.type == 'anchor' and idx < shot_cutoff_idx}

    anchor_transcripts = get_anchor_transcripts(vao, anchor_shots, 5)

    topic_to_anchor = topic_to_anchor_by_transcript(news_topics, anchor_shots, anchor_transcripts)
    topic_to_anchor = post_processing(topic_to_anchor, news_topics, anchor_shots)

    missing_topics = [idx for idx in news_topics.keys() if idx not in topic_to_anchor.keys()]
    if len(missing_topics) > 0:
        print(f'{missing_topics} could not be assigned')
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
