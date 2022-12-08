#!/Users/tihmels/Scripts/thesis-scripts/venv/bin/python -u

import logging
import os

from common.utils import set_tf_loglevel
from database import rai
from database.config import RAI_STORY_PREFIX, RAI_TEXT_PREFIX, RAI_SHOT_PREFIX

set_tf_loglevel(logging.FATAL)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from argparse import ArgumentParser

from towhee import pipeline
from alive_progress import alive_bar
from keras.applications import EfficientNetV2S
from sentence_transformers import SentenceTransformer

from database.model import MainVideo, ShortVideo, Story

IMG_ACTION = 'img'
NLP_ACTION = 'nlp'
TOPIC_ACTION = 'top'

parser = ArgumentParser('Setup RedisAI DB')
parser.add_argument('--img', dest='actions', action='append_const', const=IMG_ACTION,
                    help='Generate image embeddings for each story shot and save them to RedisAI')
parser.add_argument('--nlp', dest='actions', action='append_const', const=NLP_ACTION,
                    help='Generate sentence embeddings for each story sentence and save them to RedisAI')
parser.add_argument('--top', dest='actions', action='append_const', const=TOPIC_ACTION,
                    help='Generate sentence embeddings for each story sentence and save them to RedisAI')
parser.add_argument('--pks', action='append', type=str,
                    help='Generate sentence embeddings for each story sentence and save them to RedisAI')
parser.add_argument('--ts15', action='store_true')
parser.add_argument('--ts100', action='store_true')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')

IMAGE_SHAPE = (224, 224)

nlp_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
topic_model = SentenceTransformer('all-mpnet-base-v2')

img_model = EfficientNetV2S(weights="imagenet", include_top=False)
img_model.trainable = False


def extract_topic_features(story: Story, skip_existing):
    if skip_existing and rai.tensor_exists(RAI_STORY_PREFIX + story.pk):
        return

    feature = topic_model.encode(story.headline)

    rai.put_tensor(RAI_STORY_PREFIX + story.pk, feature)


def extract_sentence_features(story: Story, skip_existing):
    if skip_existing and all((rai.tensor_exists(RAI_TEXT_PREFIX + sentence.pk) for sentence in story.sentences)):
        return

    features = nlp_model.encode([sentence.text for sentence in story.sentences])

    for sentence, feature in zip(story.sentences, features):
        rai.put_tensor(RAI_TEXT_PREFIX + sentence.pk, feature)


def extract_image_features(story: Story, skip_existing):
    if len(story.shots) == 0:
        return

    if skip_existing and all((rai.tensor_exists(RAI_SHOT_PREFIX + shot.pk) for shot in story.shots)):
        return

    embedding_pipeline = pipeline('towhee/image-embedding-swinbase')

    embeddings = [embedding_pipeline(shot.keyframe) for shot in story.shots]

    for shot, vector in zip(story.shots, embeddings):
        rai.put_tensor(RAI_SHOT_PREFIX + shot.pk, vector)


action_map = {
    IMG_ACTION: extract_image_features,
    NLP_ACTION: extract_sentence_features,
    TOPIC_ACTION: extract_topic_features,
}


def action_dispatcher(action, video, skip_existing):
    func = action_map[action]

    for story in video.stories:
        func(story, skip_existing)


def alive_action(videos, actions, skip_existing):
    with alive_bar(len(videos),
                   ctrl_c=False,
                   title=videos[0].Meta.model_key_prefix,
                   length=50, force_tty=True) as bar:

        for video in videos:
            bar.text = video.pk

            for action in actions:
                action_dispatcher(action, video, skip_existing)

            bar()


def main(args):
    actions = args.actions

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
