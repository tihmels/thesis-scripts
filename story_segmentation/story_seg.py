#!/Users/tihmels/Scripts/thesis-scripts/venv/bin/python -u
import operator
import re
import textwrap
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from HanTa import HanoverTagger as ht
from PIL import Image
from fuzzywuzzy import fuzz
from matplotlib import pyplot as plt, patches
from nltk.corpus import stopwords
from pytesseract import pytesseract, Output
from sklearn.feature_extraction.text import CountVectorizer

from banner_ocr.ocr import crop_frame, sharpen_frame, resize_frame
from common.Schemas import STORY_COLUMNS
from common.VAO import VAO, get_shot_file, get_date_time, get_banner_caption_file, get_story_file, \
    get_topic_file, is_summary, read_shots_from_file, get_text
from common.constants import TV_FILENAME_RE
from common.utils import frame_idx_to_sec, sec_to_time, time_to_datetime, flatten
from story_segmentation.charsplit.splitter import Splitter

matplotlib.use('TkAgg')

parser = ArgumentParser('Story Segmentation')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')

spacy_de = spacy.load('de_core_news_sm')
german_stop_words = stopwords.words('german')

tagger = ht.HanoverTagger('morphmodel_ger.pgz')
splitter = Splitter()

abbrvs = {'AKW': 'Atomkraftwerk',
          'EuGH': 'Europäischer Gerichtshof',
          'EU': 'Europäische Union',
          'WM': 'Weltmeisterschaft',
          'KZ': 'Konzentrationslager'}


def lemmatizer(text):
    return [tag.lower() for word in text for tag in tagger.analyze(word)[:1]]


def extract_story_data(shots, first_shot_idx: int, last_shot_idx: int):
    n_shots = last_shot_idx - first_shot_idx + 1

    first_frame_idx = shots[first_shot_idx].first_frame_idx
    last_frame_idx = shots[last_shot_idx].last_frame_idx

    n_frames = last_frame_idx - first_frame_idx + 1

    from_time = sec_to_time(frame_idx_to_sec(first_frame_idx))
    to_time = sec_to_time(frame_idx_to_sec(last_frame_idx))

    timedelta = time_to_datetime(to_time) - time_to_datetime(from_time)
    total_ss = int(timedelta.total_seconds())

    return (first_frame_idx, last_frame_idx, n_frames, first_shot_idx, last_shot_idx, n_shots,
            from_time.strftime('%H:%M:%S'), to_time.strftime('%H:%M:%S'), total_ss)


def umlauts(text):
    text = text.replace('ä', 'ae')
    text = text.replace('ö', 'oe')
    text = text.replace('ü', 'ue')
    text = text.replace('Ä', 'Ae')
    text = text.replace('Ö', 'Oe')
    text = text.replace('Ü', 'Ue')
    text = text.replace('ß', 'ss')

    return text


def abbreviations(text):
    for abbrv, sub in abbrvs.items():
        text = text.replace(abbrv, sub)
    return text


def custom_preprocessor(text):
    text = abbreviations(text)

    text = text.lower()

    return text


def custom_tokenizer(text):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    text = token_pattern.findall(text)

    text = [word for word in text if word not in german_stop_words]

    splits = flatten([split[1:] for word in text for split in splitter.split_compound(word) if split[0] > 0.7])
    splits = [split.lower() for split in splits]

    return lemmatizer(text + splits)


def get_vectorizer(text) -> CountVectorizer:
    vectorizer = CountVectorizer(preprocessor=custom_preprocessor, tokenizer=custom_tokenizer, ngram_range=(1, 2),
                                 token_pattern=None)
    vectorizer.fit([text])

    return vectorizer


def wrap_labels(ax, width, break_long_words=True):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                                    break_long_words=break_long_words))
    ax.set_yticklabels(labels, rotation=0)


def assign_topics_to_shots(topics, shots, transcripts, texts):
    dt_matrix = np.zeros(shape=(len(topics), len(shots)))

    for idx, topic in topics.items():
        vectorizer = get_vectorizer(topic)

        voc = vectorizer.vocabulary_

        bow_caption = vectorizer.transform([caption for caption in texts.values()])
        bow_transcript = vectorizer.transform([transcript for transcript in transcripts.values()])

        bow = bow_transcript.toarray() + np.multiply(bow_caption.toarray(), 2)

        bow_sum = [sum(vec) for vec in bow]
        dt_matrix[idx] = bow_sum

    argmax, values = [], []

    for idx, topic in enumerate(topics.values()):
        lower_bound = argmax[-1] + 1 if len(argmax) > 0 else 0

        row = dt_matrix[idx, lower_bound:].astype(int)
        argmax.append(np.argmax(row) + lower_bound)
        values.append(np.max(row))

    # show_token_heatmap(argmax, dt_matrix, topics)

    to_shot_idx = np.vectorize(lambda dt_idx: list(shots.keys())[dt_idx])
    shot_idxs = to_shot_idx(argmax)

    return {topic_idx: shot_idx for idx, (topic_idx, shot_idx) in enumerate(zip(topics, shot_idxs)) if
            (values[idx] > 0 and all(pre_shot_idx < shot_idx for pre_shot_idx in shot_idxs[:idx]))}


def show_token_heatmap(argmax, dt_matrix, topics):
    ax = sns.heatmap(dt_matrix[:, :100], cmap='Blues', yticklabels=list(topics.values()),
                     cbar_kws={'label': 'Vocabulary hits', 'shrink': 0.8})
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([])
    wrap_labels(ax, 38)
    ax.set(xlabel='Shots')
    plt.subplots_adjust(left=0.2, right=0.98)
    for aidx, amax in enumerate(argmax):
        ax.add_patch(
            patches.Rectangle(
                (amax, -1),
                1.0,
                float(len(topics)) + 1,
                edgecolor='black',
                fill=False,
                lw=0.5
            ))


def get_shot_transcripts(vao: VAO, shots):
    transcripts = vao.data.transcripts

    shot_transcript_idxs = {idx: vao.data.get_shot_transcripts(idx) for idx in shots}
    return {
        shot_idx: get_text([transcripts[idx] for idx in trans_idxs]) for shot_idx, trans_idxs in
        shot_transcript_idxs.items()
    }


def confidence_text(ocr_data, threshold=80):
    positive_confidences = np.array(ocr_data['conf']) > threshold
    non_empty_texts = list(map(lambda idx: True if ocr_data['text'][idx].strip() else False,
                               range(0, len(positive_confidences))))

    positive_confidence_indices = (positive_confidences & non_empty_texts).nonzero()

    if len(positive_confidence_indices[0]) == 0:
        return ''

    strings = np.array(ocr_data['text'])[positive_confidence_indices]

    return re.sub("\\W", " ", ' '.join(strings))


def get_caption(frame: Path, resize_factor=3):
    frame = Image.open(frame).convert('L')

    area = (55, 157, 280, 223)
    frame = crop_frame(frame, area)

    frame = sharpen_frame(frame, 2)

    frame = resize_frame(frame, resize_factor)

    frame = np.array(frame) > 160

    news_text_data = pytesseract.image_to_data(frame, output_type=Output.DICT, lang='deu',
                                               config='--psm 6 --oem 1')

    return confidence_text(news_text_data)


def get_shot_text(vao: VAO, anchor_shots):
    return {
        idx: get_caption(vao.data.frames[shot.center_frame_idx]) for idx, shot in anchor_shots.items()
    }


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

    shots = {idx: sd for idx, sd in enumerate(vao.data.shots) if idx < shot_cutoff_idx}

    transcripts = get_shot_transcripts(vao, shots)
    banner_texts = get_shot_text(vao, shots)

    topic_to_shot = assign_topics_to_shots(news_topics, shots, transcripts, banner_texts)
    topic_to_shot = post_processing(topic_to_shot, news_topics, shots)

    missing_topics = [idx for idx in news_topics.keys() if idx not in topic_to_shot.keys()]
    if len(missing_topics) > 0:
        print(f'{missing_topics} could not be assigned')
    else:
        print()

    stories = []

    for topic_idx, anchor_idx in topic_to_shot.items():
        story_title = news_topics[topic_idx]

        first_shot_idx = anchor_idx

        if topic_idx + 1 in list(topic_to_shot.keys()):
            last_shot_idx = next((idx for idx in list(topic_to_shot.values()) if idx > first_shot_idx)) - 1
        else:
            last_shot_idx = next(
                (idx for idx, cd in shots.items() if idx > first_shot_idx and cd.type == 'anchor'),
                len(shots)) - 1

        story_data = extract_story_data(vao.data.shots, first_shot_idx, last_shot_idx)

        stories.append((topic_idx, story_title, *story_data))

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
        if (not cd.text.strip() or cd.confidence < 70) and idx - 1 in captions:
            predecessor = captions[idx - 1]
            captions[idx] = predecessor


def get_story_indices_by_captions(captions):
    first_idx, last_idx = min(captions.keys()), max(captions.keys())
    current_caption = captions[first_idx].text

    story_indices = [[first_idx]]

    for idx in range(first_idx + 1, last_idx):
        next_caption = captions[idx].text

        ratio = fuzz.token_set_ratio(current_caption, next_caption)

        if ratio > 80:
            story_indices[-1].append(idx)
        else:
            story_indices.append([idx])

        current_caption = next_caption

    return story_indices


def segment_ts100(vao: VAO):
    captions = {idx: cd for idx, cd in enumerate(vao.data.banners[:-1])}  # last shot is always weather

    preprocess_captions(captions)

    story_indices = get_story_indices_by_captions(captions)

    stories = []

    for topic_idx, story in enumerate(story_indices):
        story_captions = {idx: cd for idx, cd in captions.items() if idx in story}

        if story[0] == story[-1] and (
                list(story_captions.values())[0].text == "" or len(list(story_captions.values())[0].text.split()) < 3):
            continue

        max_confidence = max([caption.confidence for caption in story_captions.values()])
        story_caption = next((caption for caption in story_captions.items() if caption[1].confidence == max_confidence))

        story_data = extract_story_data(vao.data.shots, min(story), max(story))

        stories.append((story_caption[0], story_caption[1].text, *story_data))

    df = pd.DataFrame(data=stories, columns=STORY_COLUMNS)

    print()

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

    n_total = 0
    unassigned_topics = []

    for idx, vf in enumerate(video_files):
        vao = VAO(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vao} ', end='... ')

        segmentor = segment_ts100 if vao.is_summary else segment_ts15

        df = segmentor(vao)

        if not vao.is_summary:
            assigned_topics_idxs = df.loc[:, 'ref_idx'].astype(int).tolist()

            topics_cutoff_idx = next(
                idx for idx, topic in enumerate(vao.data.topics) if re.search(r"^(Die Lottozahlen|Das Wetter)$", topic))

            n_topics = topics_cutoff_idx
            n_total += n_topics

            unassigned_topics.extend(
                [topic for idx, topic in enumerate(vao.data.topics[:topics_cutoff_idx]) if
                 idx not in assigned_topics_idxs])

        df.to_csv(get_story_file(vao), index=False)

    print(f'{n_total - len(unassigned_topics)} / {n_total} topics could be assigned.')
    print(f'unassigned topics are: ${unassigned_topics}')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
