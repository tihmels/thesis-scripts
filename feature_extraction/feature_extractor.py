#!/Users/tihmels/Scripts/thesis-scripts/venv/bin/python -u

import os
import torch
from redis_om import Migrator
from sentence_transformers import SentenceTransformer

from database import rai
from database.config import RAI_TOPIC_PREFIX, get_vis_key, \
    get_m5c_key, get_text_key, get_topic_key
from feature_extraction.StoryDataExtractor import StoryDataExtractor

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_hub as hub

from alive_progress import alive_bar

from database.model import MainVideo, ShortVideo, Story, TopicCluster

TOPIC_ACTION = 'top'
MIL_NCE_ACTION = 'mil'

parser = ArgumentParser('Setup RedisAI DB')
parser.add_argument('--top', dest='actions', action='append_const', const=TOPIC_ACTION,
                    help='Generate sentence embeddings for each story headline and save them to RedisAI')
parser.add_argument('--mil', dest='actions', action='append_const', const=MIL_NCE_ACTION,
                    help='Generate sentence embeddings for each story sentence and save them to RedisAI')
parser.add_argument('--pks', action='append', type=str,
                    help='Generate sentence embeddings for each story sentence and save them to RedisAI')
parser.add_argument('--ts15', action='store_true')
parser.add_argument('--ts100', action='store_true')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')

# topic_model = SentenceTransformer('all-mpnet-base-v2')

mil_nce_module = 'https://tfhub.dev/deepmind/mil-nce/s3d/1'
mil_nce = hub.load(mil_nce_module)


def extract_milnce_features(stories: [Story], dataset, skip_existing):
    extractor = StoryDataExtractor(stories, dataset)

    with torch.no_grad():
        with alive_bar(len(extractor), ctrl_c=False, title=f'MIL-NCE [{dataset}]', length=25, dual_line=True,
                       receipt_text=True) as bar:
            for i in range(len(extractor)):
                story_pk = stories[i].pk

                bar.text = f'Story: {story_pk}'

                if skip_existing and rai.tensor_exists(get_vis_key(story_pk)):
                    bar()
                    continue

                segments, sentences = extractor[i]

                if len(segments) > 0:
                    vision_output = mil_nce.signatures['video'](tf.constant(tf.cast(segments, dtype=tf.float32)))
                    segment_features = vision_output['video_embedding'].numpy()
                    mixed_5c = vision_output['mixed_5c'].numpy()

                    rai.put_tensor(get_vis_key(story_pk), segment_features)
                    rai.put_tensor(get_m5c_key(story_pk), mixed_5c)

                    if len(sentences) > 0:
                        text_output = mil_nce.signatures['text'](tf.constant(sentences))
                        text_features = text_output['text_embedding'].numpy()

                        rai.put_tensor(get_text_key(story_pk), text_features)

                bar()


def extract_topic_features(story: Story, skip_existing):
    if skip_existing and rai.tensor_exists(get_topic_key(story.pk)):
        return

    feature = topic_model.encode(story.headline)

    rai.put_tensor(RAI_TOPIC_PREFIX + story.pk, feature)


action_map = {
    TOPIC_ACTION: extract_topic_features,
    MIL_NCE_ACTION: extract_milnce_features
}


def action_dispatcher(action, video, skip_existing):
    func = action_map[action]

    for story in video.stories:
        func(story, skip_existing)


def alive_action(videos, actions, skip_existing):
    with alive_bar(len(videos),
                   ctrl_c=False,
                   title=videos[0].Meta.model_key_prefix,
                   length=25, force_tty=True) as bar:

        for video in videos:
            bar.text = video.pk

            for action in actions:
                action_dispatcher(action, video, skip_existing)

            bar()


def main(args):
    actions = args.actions

    if MIL_NCE_ACTION in actions:
        clusters = TopicCluster.find().sort_by('-index').all()

        if not args.skip_existing:
            for cluster in clusters:
                cluster.features = 0
                cluster.save()

        for idx, cluster in enumerate(clusters):
            print(f'[{idx + 1}/{len(clusters)}] Cluster: {cluster.index}')
            extract_milnce_features(cluster.ts15s, dataset='ts15', skip_existing=args.skip_existing)
            extract_milnce_features(cluster.ts100s, dataset='ts100', skip_existing=args.skip_existing)

            cluster.features = 1
            cluster.save()

            Migrator().run()

    if args.pks:
        videos = [MainVideo.get(pk) for pk in args.pks]
        alive_action(videos, actions, args.skip_existing)

    if args.ts15:
        videos = MainVideo.find().sort_by('timestamp').all()
        alive_action(videos, actions, args.skip_existing)

    if args.ts100:
        videos = ShortVideo.find().sort_by('timestamp').all()
        alive_action(videos, actions, args.skip_existing)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
