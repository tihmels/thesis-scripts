import cv2
import numpy as np
import torchvision.transforms as transforms

from common.utils import read_images
from database.model import Story

IMAGE_SHAPE = (224, 224)


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_frames(frame_paths, resize=IMAGE_SHAPE):
    frames = [crop_center_square(frame) for frame in read_images(frame_paths)]
    frames = [cv2.resize(frame, resize) for frame in frames]

    frames = np.array(frames)

    return frames / 255.0


class StoryDataExtractor:
    def __init__(self, stories: [Story], window=1, dataset='ts15'):
        self.stories = stories
        self.window = window
        self.dataset = dataset
        self.transform = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]

        frames = load_frames(story.frames[::5])

        window_len = 32

        while len(frames) % window_len != 0:
            extra_frames = window_len - (len(frames) % window_len)
            frames = np.concatenate((frames, frames[-extra_frames:]), axis=0)

        n_segments = int(frames.shape[0] / window_len)

        segments = np.reshape(frames, (n_segments, window_len, frames.shape[1], frames.shape[2], 3))

        return story.pk, segments, story.sentences
