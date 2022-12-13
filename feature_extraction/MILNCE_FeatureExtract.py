import os
import torch
import torchvision.io as io
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from common.utils import read_images
from database.model import Story


class MilNceFeatureExtractor(Dataset):
    def __init__(self, stories: [Story], window=1, dataset='ts15'):
        self.stories = stories
        self.window = window
        self.dataset = dataset

        self.transform = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        frames = read_images(story.frames[::5])
        frames = torch.tensor(frames)

        window_len = int(self.window * 25)

        # Pad video with extra frames to ensure its divisible by window_len
        extra_frames = window_len - (len(frames) % window_len)
        video = torch.cat((frames, frames[-extra_frames:]), dim=0)

        n_segs = int(video.shape[0] / window_len)

        print("Number of video segments: ", n_segs)

        video = video.view(n_segs, 32, video.shape[1], video.shape[2], 3)
        video = video.permute(0, 1, 4, 2, 3)

        # Transform video segments
        video_segs = []
        for seg in video:
            # Resize and normalize to [0,1]
            video_segs.append(self.transform(seg) / 255.0)
        video_segs = torch.stack(video_segs)
        video_segs = video_segs.view(n_segs, 32, 3, 224, 224)

        return video_segs, self.vid_files[idx].split(".")[0]
