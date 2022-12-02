#!/Users/tihmels/Scripts/thesis-scripts/venv/bin/python -u
import logging

from common.utils import set_tf_loglevel

set_tf_loglevel(logging.FATAL)

from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from alive_progress import alive_bar
from keras.applications import EfficientNetV2S
from sentence_transformers import SentenceTransformer
from smartredis import Client

from database.model import MainVideo, ShortVideo

IMG_ACTION = 'img'
NLP_ACTION = 'nlp'

parser = ArgumentParser('Setup RedisAI DB')
parser.add_argument('--img', dest='actions', action='append_const', const=IMG_ACTION,
                    help='Generate image embeddings for each story shot and save them to RedisAI')
parser.add_argument('--nlp', dest='actions', action='append_const', const=NLP_ACTION,
                    help='Generate sentence embeddings for each story sentence and save them to RedisAI')
parser.add_argument('--pks', action='append', type=str,
                    help='Generate sentence embeddings for each story sentence and save them to RedisAI')
parser.add_argument('--ts15', action='store_true')
parser.add_argument('--ts100', action='store_true')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')

IMAGE_SHAPE = (224, 224)

nlp_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

db_address = "localhost:6379"
rai = Client(address=db_address)

img_model = EfficientNetV2S(weights="imagenet", include_top=False)
img_model.trainable = False


def extract_sentence_features(story, skip_existing):
    if skip_existing and all((rai.tensor_exists(sentence.pk) for sentence in story.sentences)):
        return

    features = nlp_model.encode([sentence.text for sentence in story.sentences])

    for sentence, feature in zip(story.sentences, features):
        rai.put_tensor(sentence.pk, feature)


def extract_image_features(story, skip_existing):
    if len(story.shots) == 0:
        return

    if skip_existing and all((rai.tensor_exists(shot.pk) for shot in story.shots)):
        return

    frames = [tf.keras.utils.load_img(shot.keyframe, target_size=IMAGE_SHAPE) for shot in story.shots]
    frames = [tf.keras.utils.img_to_array(frame) for frame in frames]
    frames = np.expand_dims(frames, axis=0)

    features = img_model.predict(np.vstack(frames), verbose=0)

    for shot, feature in zip(story.shots, features):
        rai.put_tensor(shot.pk, feature.flatten())


def action_dispatcher(action, video, skip_existing):
    func = extract_image_features if action == IMG_ACTION else extract_sentence_features

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
        videos = [ShortVideo.get(pk) for pk in args.pks]
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
