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
import torch
from PIL import Image
from redis_om import Migrator

from database.model import MainVideo

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
checkpoint = torch.load("./model_checkpoint/model_10.pt")


# Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"]
# checkpoint['model_state_dict']["input_resolution"] = 224  # default is 224
# checkpoint['model_state_dict']["context_length"] = 77  # default is 77
# checkpoint['model_state_dict']["vocab_size"] = model.vocab_size

# model.load_state_dict(checkpoint['model_state_dict'])

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def create_summary(video: MainVideo, query):
    stories = video.stories

    story_cosines = []

    text = clip.tokenize(query).to(device)

    with torch.no_grad():
        for story in stories:
            print(story.headline)

            text_features = model.encode_text(text)

            images = [preprocess(Image.open(shot.keyframe)).unsqueeze(0).to(device) for shot in story.shots]
            image_features = [model.encode_image(image) for image in images]

            cos = torch.nn.CosineSimilarity(dim=1)

            cosine_distances = [cos(image, text_features) for image in image_features]

            print(f'Cosine Similarities: {cosine_distances}')

            mean = torch.mean(torch.stack(cosine_distances))
            print('Mean: ' + str(mean))

            print()

    pass


def main():
    video = MainVideo.find().sort_by('timestamp').first()

    create_summary(video, query='No politicians')

    Migrator().run()


if __name__ == "__main__":
    main()
