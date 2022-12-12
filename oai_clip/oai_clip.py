#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

# Latest Update : 18 July 2022, 09:55 GMT+7

# TO ADD :
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used

# BATCH_SIZE must larger than 1
import clip
import random
import torch
from PIL import Image
from alive_progress import alive_bar
from redis_om import Migrator
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from clustering.clustering import process_stories
from database import rai
from database.config import RAI_TOPIC_PREFIX
from database.model import MainVideo, Story, ShortVideo

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training

EPOCHS = 10
BATCH_SIZE = 64


class ImageTextDataset(Dataset):
    def __init__(self):
        self.image_paths = []
        self.transcripts = None

    def add(self, image_paths, image_texts):
        self.image_paths.append(image_paths)
        if self.transcripts is None:
            self.transcripts = clip.tokenize(image_texts, truncate=True)
        else:
            tensor = clip.tokenize(image_texts, truncate=True)
            self.transcripts = torch.cat((self.transcripts, tensor))

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_paths[idx]))  # Image from PIL module
        title = self.transcripts[idx]
        return image, title


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def create_data_loader(ts15_stories, ts100_stories, dataset):
    data = []

    for story in ts15_stories:

        shots = story.shots[1:] if story.shots[0].type == 'anchor' else story.shots

        for shot in shots:
            data.append((shot.keyframe, shot.transcript))

    for story in ts100_stories:

        for shot in story.shots:
            data.append((shot.keyframe, shot.transcript))

    random.shuffle(data)

    for keyframe, transcript in data:
        dataset.add(keyframe, transcript)

    return DataLoader(dataset, batch_size=BATCH_SIZE)  # Define your own dataloader


def process_stories(ts15_stories: [Story], ts100_stories: [Story]):
    dataset = ImageTextDataset()

    dataloader = create_data_loader(random.sample(ts15_stories, 100), random.sample(ts100_stories, 200), dataset)

    if device == "cpu":
        model.float()

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    with alive_bar(EPOCHS, title=f'Training', length=25, dual_line=True) as bar:
        for epoch in range(EPOCHS):
            for idx, batch in enumerate(dataloader):

                optimizer.zero_grad()

                images, texts = batch

                images = images.to(device)
                texts = texts.to(device)

                logits_per_image, logits_per_text = model(images, texts)

                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

                total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                total_loss.backward()

                if device == "cpu":
                    optimizer.step()

                bar.text = f'Batch {idx + 1}/{int(len(dataloader.dataset)/BATCH_SIZE)} - Total Loss: {total_loss}'

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                }, f"model_checkpoint/model_10.pt")
            bar()


def main():
    ts15_videos = MainVideo.find().sort_by('timestamp').all()
    ts100_videos = ShortVideo.find().sort_by('timestamp').all()

    assert len(ts15_videos) > 0, 'No suitable video files have been found.'

    ts15_stories = [story for video in ts15_videos for story in video.stories if
                    rai.tensor_exists(RAI_TOPIC_PREFIX + story.pk)]
    ts100_stories = [story for video in ts100_videos for story in video.stories if
                     rai.tensor_exists(RAI_TOPIC_PREFIX + story.pk)]

    process_stories(ts15_stories, ts100_stories)

    Migrator().run()


if __name__ == "__main__":
    main()
