import cv2
import numpy as np
from libretranslatepy import LibreTranslateAPI

from common.utils import read_images
from database.model import Story

IMAGE_SHAPE = (224, 224)


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


# Maybe try for optimal resizing: https://github.com/sayakpaul/Learnable-Image-Resizing
def load_frames(frame_paths, dataset, resize=IMAGE_SHAPE):
    frames = [frame for frame in read_images(frame_paths)]
    frames = [frame[:224, :] for frame in frames]
    frames = [cv2.resize(frame, resize, interpolation=cv2.INTER_AREA) for frame in frames]

    return np.array(frames) / 255.0


class StoryDataExtractor:
    def __init__(self, stories: [Story], dataset, window=24):
        self.stories = stories
        self.window = window
        self.dataset = dataset
        self.lt = LibreTranslateAPI("http://127.0.0.1:5005")

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]

        frames = load_frames(story.frames[::2], self.dataset)

        window_len = self.window

        while len(frames) % window_len != 0:
            extra_frames = window_len - (len(frames) % window_len)
            if extra_frames / window_len > 0.5:
                frames = frames[:-(window_len - extra_frames)]
            else:
                frames = np.concatenate((frames, frames[-extra_frames:]), axis=0)

        n_segments = int(frames.shape[0] / window_len)

        segments = np.reshape(frames, (n_segments, window_len, frames.shape[1], frames.shape[2], 3))

        # sentences = [self.lt.translate(sentence, 'de', 'en') for sentence in story.sentences]
        sentences = []

        return segments, sentences
