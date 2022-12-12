# !/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

# Latest Update : 18 July 2022, 09:55 GMT+7

# TO ADD :
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used

# BATCH_SIZE must larger than 1
import clip
import numpy as np
import torch
from PIL import Image
from redis_om import Migrator

from numpy.linalg import norm

from database.model import MainVideo

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

# model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
#checkpoint = torch.load("./model_checkpoint/model_10.pt")


# Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"]
# checkpoint['model_state_dict']["input_resolution"] = 224  # default is 224
# checkpoint['model_state_dict']["context_length"] = 77  # default is 77
# checkpoint['model_state_dict']["vocab_size"] = model.vocab_size

# model.load_state_dict(checkpoint['model_state_dict'])

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

from sentence_transformers import SentenceTransformer, util
image_model = SentenceTransformer('clip-ViT-B-32')
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')


def create_summary(video: MainVideo, query):
    stories = video.stories

    story_cosines = []

    text_embedding = text_model.encode(query)

    with torch.no_grad():
        for story in stories:
            print(story.headline)

            keyframes = [Image.open(shot.keyframe) for shot in story.shots]

            image_embeddings = image_model.encode(keyframes)

            cosine_distances = [util.cos_sim(image, text_embedding) for image in image_embeddings]

            max_index = torch.argmax(torch.stack(cosine_distances))
            min_index = torch.argmin(torch.stack(cosine_distances))
            mean = torch.mean(torch.stack(cosine_distances))

            print('Query: ' + str(query))
            print('Min: ' + str(cosine_distances[min_index]) + ' - ' + str(story.shots[min_index].keyframe))
            print('Max: ' + str(cosine_distances[max_index]) + ' - ' + str(story.shots[max_index].keyframe))
            print('Mean: ' + str(mean))

            print()

    pass


def main():
    video = MainVideo.find().sort_by('timestamp').all()[1]

    create_summary(video, query='')

    Migrator().run()


if __name__ == "__main__":
    main()
