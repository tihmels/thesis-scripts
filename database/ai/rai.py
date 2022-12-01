import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from smartredis import Client
from tensorflow.keras.applications import EfficientNetB0

from database.model import MainVideo

IMG_SIZE = 224

db_address = "localhost:6379"
client = Client(address=db_address)


def extract_nlp_tensors(story):
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    sentence_embeddings = model.encode([sentence.text for sentence in story.sentences])

    for sentence, embedding in zip(story.sentences, sentence_embeddings):
        client.put_tensor(sentence.pk, embedding)


def extract_image_tensors(story):
    if len(story.shots) == 0:
        return

    model = EfficientNetB0(include_top=False, weights='imagenet')

    frames = [tf.keras.utils.load_img(shot.keyframe, target_size=(IMG_SIZE, IMG_SIZE)) for shot in
              story.shots]
    frames = [tf.keras.utils.img_to_array(frame) for frame in frames]
    frames = np.expand_dims(frames, axis=0)

    predictions = model.predict(np.vstack(frames))

    for shot, embedding in zip(story.shots, predictions):
        client.put_tensor(shot.pk, embedding)


def main():
    ts15_pks = list(MainVideo.all_pks())

    for pk in ts15_pks:
        stories = MainVideo.get(pk).stories

        for story in stories:
            extract_image_tensors(story)
            #extract_nlp_tensors(story)


if __name__ == "__main__":
    main()
