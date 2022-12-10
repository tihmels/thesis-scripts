#!/Users/tihmels/Scripts/thesis-scripts/venv/bin/python -u
import cv2
import logging
import math
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub

from common.utils import set_tf_loglevel, read_images, crop_center_square
from database import rai, db
from database.config import RAI_TOPIC_PREFIX, RAI_TEXT_PREFIX, RAI_SHOT_PREFIX, RAI_STORY_PREFIX

set_tf_loglevel(logging.FATAL)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from argparse import ArgumentParser
from ulid import ULID

from towhee import pipeline
from alive_progress import alive_bar
from keras.applications import EfficientNetV2S
from sentence_transformers import SentenceTransformer

from database.model import MainVideo, ShortVideo, Story, TopicCluster

IMG_ACTION = 'img'
NLP_ACTION = 'nlp'
TOPIC_ACTION = 'top'
VIDEO_ACTION = 'vid'

parser = ArgumentParser('Setup RedisAI DB')
parser.add_argument('--img', dest='actions', action='append_const', const=IMG_ACTION,
                    help='Generate image embeddings for each story shot and save them to RedisAI')
parser.add_argument('--nlp', dest='actions', action='append_const', const=NLP_ACTION,
                    help='Generate sentence embeddings for each story sentence and save them to RedisAI')
parser.add_argument('--top', dest='actions', action='append_const', const=TOPIC_ACTION,
                    help='Generate sentence embeddings for each story sentence and save them to RedisAI')
parser.add_argument('--vid', dest='actions', action='append_const', const=VIDEO_ACTION,
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

hub_handle = 'https://tfhub.dev/deepmind/mil-nce/s3d/1'
hub_model = hub.load(hub_handle)


def load_video(frame_paths, max_frames=32, resize=(224, 224)):
    frames = [crop_center_square(frame) for frame in read_images(frame_paths[:max_frames])]
    frames = [cv2.resize(frame, resize) for frame in frames]
    frames = [frame[:, :, [2, 1, 0]] for frame in frames]

    frames = np.array(frames)

    if len(frames) < max_frames:
        n_repeat = int(math.ceil(max_frames / float(len(frames))))
        frames = frames.repeat(n_repeat, axis=0)

    frames = frames[:max_frames]
    return frames / 255.0


def extract_video_text_features(stories: [Story], skip_existing):
    """Generate embeddings from the model from video frames and input words."""

    if skip_existing and all([db.get_key(RAI_STORY_PREFIX + story.pk) for story in stories]):
        return

    for story in stories:
        frames = load_video(story.frames)
        words = ' '.join(story.sentences)
        vision_output = hub_model.signatures['video'](tf.constant(tf.cast(frames, dtype=tf.float32)))
        text_output = hub_model.signatures['text'](tf.constant(words))

        features = []

        zset = db.ZSet(RAI_STORY_PREFIX + story.pk)
        for i in range(len(features)):
            uuid = str(ULID())
            rai.put_tensor(RAI_STORY_PREFIX + uuid, features[i])
            zset.add({uuid: i})


def extract_topic_features(story: Story, skip_existing):
    if skip_existing and rai.tensor_exists(RAI_TOPIC_PREFIX + story.pk):
        return

    feature = topic_model.encode(story.headline)

    rai.put_tensor(RAI_TOPIC_PREFIX + story.pk, feature)


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
    VIDEO_ACTION: extract_video_text_features
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

    if VIDEO_ACTION in actions:
        clusters = TopicCluster.find().all()

        for cluster in clusters:
            stories = cluster.stories
            extract_video_text_features(stories, args.skip_existing)

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
