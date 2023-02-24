import cv2
import numpy as np

from common.utils import read_images
from database.model import Story

IMAGE_SHAPE = (224, 224)


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_frames(frame_paths, type, base_path, resize=IMAGE_SHAPE):
    frames = read_images(frame_paths, base_path=base_path)

    if type == 'ts100':
        frames = [frame[:224, :] for frame in frames]

    frames = [cv2.resize(frame, resize, interpolation=cv2.INTER_AREA) for frame in frames]

    return np.array(frames) / 255.0


class StoryDataExtractor:
    def __init__(self, skip_n=3, window=16, base_path='/Users/tihmels/TS'):
        self.skip_n = skip_n
        self.window = window
        self.base_path = base_path

    def extract_data(self, story: Story):
        frames = load_frames(story.frames[::self.skip_n], story.type, self.base_path)

        window_len = self.window

        while len(frames) % window_len != 0:
            extra_frames = window_len - (len(frames) % window_len)
            if extra_frames / window_len > 0.5:
                frames = frames[:-(window_len - extra_frames)]
            else:
                frames = np.concatenate((frames, frames[-extra_frames:]), axis=0)

        n_segments = int(frames.shape[0] / window_len)

        segments = np.reshape(frames, (n_segments, window_len, frames.shape[1], frames.shape[2], 3))

        return segments, story.sentences
