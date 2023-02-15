import math
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from common.utils import read_images, flatten
from database import db, rai
from database.model import Story, MainVideo, get_sum_key, get_score_key


class NewsSumStoryLoader(Dataset):
    """TVSum_DataLoader Video loader."""

    def __init__(
            self,
            fps=8,
            num_frames=480,
            num_frames_per_segment=16,
            size=224
    ):
        assert isinstance(size, int)

        self.fps = fps
        self.num_frames = num_frames
        self.num_sec = self.num_frames / float(self.fps)
        self.num_frames_per_segment = num_frames_per_segment
        self.n_segments = int(self.num_frames / self.num_frames_per_segment)
        self.size = size

        self.stories = Story.find(Story.type == 'ts15').all()
        self.stories = [story for story in self.stories if db.List(get_sum_key(story.pk))]

        self.summaries = [db.List(get_sum_key(story.pk)).as_list() for story in self.stories]
        self.summaries = [list(map(int, map(float, label))) for label in self.summaries]

        self.scores = [db.List(get_score_key(story.pk)).as_list() for story in self.stories]
        self.scores = [list(map(float, score)) for score in self.scores]

        pos = 0
        neg = 0

        pos += sum([np.count_nonzero(np.asarray(label)) for label in self.summaries])
        neg += sum(len(label) - np.count_nonzero(np.asarray(label)) for label in self.summaries)

        print("Pos neg: ", pos, neg)
        self.ce_weight = torch.tensor(
            [(pos + neg) / neg, (pos + neg) / pos], dtype=torch.float32
        )

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]

        print(f'Loading Story: {story.pk} ...')

        summary = torch.LongTensor(self.summaries[idx])
        scores = torch.FloatTensor(self.scores[idx])
        # scores = torch.FloatTensor(self.scores[idx]) / 0.1

        n_story_segments = len(summary)

        first_segment_idx = random.randint(0, int(max(0, n_story_segments - self.n_segments)))
        last_segment_idx = min(n_story_segments, first_segment_idx + self.n_segments)

        skip_n = math.floor(25 / self.fps)

        first_frame_idx = first_segment_idx * skip_n * self.num_frames_per_segment
        last_frame_idx = min(last_segment_idx * skip_n * self.num_frames_per_segment, len(story.frames) - 1)

        frames = read_images(story.frames[first_frame_idx:last_frame_idx:3])

        summary = summary[first_segment_idx:last_segment_idx]
        scores = scores[first_segment_idx:last_segment_idx]

        if summary.shape[0] < self.n_segments:
            summary = torch.cat(
                (summary, torch.zeros((self.n_segments - len(summary)), dtype=torch.long))
            )
            scores = torch.cat(
                (scores, torch.zeros((self.n_segments - len(scores))))
            )

        assert summary.shape[0] == self.n_segments
        assert scores.shape[0] == self.n_segments

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((self.size, self.size))]
        )

        video = torch.stack([transform(frame) for frame in frames])

        video = video.view(-1, 3, self.size, self.size)

        video = video.permute(1, 0, 2, 3)

        if video.shape[1] < self.num_frames:
            zeros = torch.zeros(
                (3, self.num_frames - video.shape[1], self.size, self.size),
                dtype=torch.uint8,
            )
            video = torch.cat((video, zeros), axis=1)

        video = video[:, :self.num_frames]

        return {"pk": story.pk, "video": video, "summary": summary, "scores": scores}
