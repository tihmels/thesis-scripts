#!/Users/tihmels/Scripts/thesis-scripts/venv/bin/python -u
import logging
import os
import sys
import torch
from alive_progress import alive_bar
from argparse import ArgumentParser
from redis_om import Migrator

from common.utils import set_tf_loglevel
from database import rai
from database.model import MainVideo, ShortVideo, Story, get_vis_key, get_m5c_key, get_topic_key, get_text_key
from feature_extraction.StoryDataExtractor import StoryDataExtractor

set_tf_loglevel(logging.FATAL)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import tensorflow as tf
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer

TOPIC_ACTION = 'top'
MIL_NCE_ACTION = 'mil'

parser = ArgumentParser('Setup RedisAI DB')
parser.add_argument('--top', '--topic', dest='actions', action='append_const', const=TOPIC_ACTION,
                    help='Generate sentence embeddings for each story headline and save them to RedisAI')
parser.add_argument('--mil', dest='actions', action='append_const', const=MIL_NCE_ACTION,
                    help='Generate sentence embeddings for each story sentence and save them to RedisAI')
parser.add_argument('--ts15', action='store_true')
parser.add_argument('--ts100', action='store_true')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')

print('Loading Models ...')

multi_mpnet_model = 'paraphrase-multilingual-mpnet-base-v2'
t_systems_model = 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
embedder = SentenceTransformer(multi_mpnet_model)

mil_nce_model = 'https://tfhub.dev/deepmind/mil-nce/s3d/1'
mil_nce = hub.load(mil_nce_model)


def extract_milnce_features(story: Story, skip_existing):
    if skip_existing and rai.tensor_exists(get_vis_key(story.pk)):
        return

    extractor = StoryDataExtractor()

    with torch.no_grad():

        segments, sentences = extractor.extract_data(story)

        if len(segments) > 0:
            vision_output = mil_nce.signatures['video'](tf.constant(tf.cast(segments, dtype=tf.float32)))
            segment_features = vision_output['video_embedding'].numpy()
            mixed_5c = vision_output['mixed_5c'].numpy()

            rai.put_tensor(get_vis_key(story.pk), segment_features)
            rai.put_tensor(get_m5c_key(story.pk), mixed_5c)

            if len(sentences) > 0:
                text_output = mil_nce.signatures['text'](tf.constant(sentences))
                text_features = text_output['text_embedding'].numpy()

                rai.put_tensor(get_text_key(story.pk), text_features)


def extract_topic_features(story: Story, skip_existing):
    if skip_existing and rai.tensor_exists(get_topic_key(story.pk)):
        return

    embedding = embedder.encode(story.headline, convert_to_numpy=True)

    rai.put_tensor(get_topic_key(story.pk), embedding)


action_map = {
    TOPIC_ACTION: ('Topic Embedding', extract_topic_features),
    MIL_NCE_ACTION: ('MIL-NCE', extract_milnce_features)
}


def alive_action(videos, actions, skip_existing):
    for action in actions:

        title, handler = action_map[action]

        with alive_bar(len(videos), ctrl_c=False, title=title, length=25, dual_line=True) as bar:
            for video in videos:

                for idx, story in enumerate(video.stories):
                    bar.text = f'[{video.Meta.model_key_prefix}] {story.pk}'
                    handler(story, skip_existing)

                bar()


def extract_targets(args):
    ts15 = []
    ts100 = []

    if args.ts15:
        ts15 = MainVideo.find().sort_by('timestamp').all()

    if args.ts100:
        ts100 = ShortVideo.find().sort_by('timestamp').all()

    return sorted(ts15 + ts100, key=lambda t: t.timestamp)


def main(args):
    actions = args.actions

    print("Extracting Targets ...")
    targets = extract_targets(args)

    alive_action(targets, actions, args.skip_existing)

    Migrator().run()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

    sys.exit()
