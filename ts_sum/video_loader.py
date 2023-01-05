import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from common.utils import read_images, flatten
from database import db, rai
from database.config import get_sum_key, get_score_key
from database.model import Story, MainVideo


class TVSumStoryLoader(Dataset):
    """VSum_DataLoader Video loader."""

    def __init__(
            self,
            fps=12,
            num_frames=832,
            num_frames_per_segment=32,
            size=224,
            token_to_word_path="data/dict.npy",
    ):
        assert isinstance(size, int)

        self.fps = fps
        self.num_frames = num_frames
        self.num_sec = self.num_frames / float(self.fps)
        self.num_frames_per_segment = num_frames_per_segment
        self.size = size

        self.stories = Story.find(Story.type == 'ts15').all()
        self.stories = [story for story in self.stories if db.List(get_sum_key(story.pk))]
        self.stories = self.stories[:12]

        token_to_word = np.load(
            os.path.join(os.path.dirname(__file__), token_to_word_path)
        )

        self.word_to_token = {}
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

        pos = 0
        neg = 0

        self.summaries = [db.List(get_sum_key(story.pk)).as_list() for story in self.stories]
        self.summaries = [list(map(int, map(float, label))) for label in self.summaries]

        self.scores = [db.List(get_score_key(story.pk)).as_list() for story in self.stories]
        self.scores = [list(map(float, score)) for score in self.scores]

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
        scores = torch.FloatTensor(self.scores[idx]) / 0.1

        frames = read_images(story.frames[::2])

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

        n_segments = int(self.num_frames / self.num_frames_per_segment)

        start_frame = 0

        summary = summary[start_frame: start_frame + n_segments]
        scores = scores[start_frame: start_frame + n_segments]

        if summary.shape[0] < n_segments:
            summary = torch.cat(
                (summary, torch.zeros((n_segments - len(summary)), dtype=torch.long))
            )
            scores = torch.cat(
                (scores, torch.zeros((n_segments - len(scores))))
            )

        summary = torch.LongTensor(summary)
        assert summary.shape[0] == n_segments
        assert scores.shape[0] == n_segments

        return {"video": video, "summary": summary, "scores": scores}


class TVSumVideoLoader(Dataset):
    """VSum_DataLoader Video loader."""

    def __init__(
            self,
            fps=12,
            num_frames=10880,
            num_frames_per_segment=32,
            size=224,
            token_to_word_path="data/dict.npy",
    ):
        assert isinstance(size, int)

        self.fps = fps
        self.num_frames = num_frames
        self.num_sec = self.num_frames / float(self.fps)
        self.num_frames_per_segment = num_frames_per_segment
        self.size = size

        self.videos = MainVideo.find().all()
        self.videos = [video for video in self.videos if
                       all([rai.tensor_exists(get_sum_key(story.pk)) for story in video.stories])]

        token_to_word = np.load(
            os.path.join(os.path.dirname(__file__), token_to_word_path)
        )

        self.word_to_token = {}
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

        pos = 0
        neg = 0

        video_stories = [video.stories for video in self.videos]

        self.summaries = []
        for stories in video_stories:
            summaries = [db.List(get_sum_key(story.pk)).as_list() for story in stories]
            summaries = [list(map(int, map(float, label))) for label in summaries]
            self.summaries.append(summaries)

        self.scores = []
        for stories in video_stories:
            scores = [db.List(get_score_key(story.pk)).as_list() for story in stories]
            scores = [list(map(float, score)) for score in scores]
            self.scores.append(scores)

        pos += sum([np.count_nonzero(np.asarray(label)) for label in flatten(self.summaries)])
        neg += sum(len(label) - np.count_nonzero(np.asarray(label)) for label in flatten(self.summaries))

        print("Pos neg: ", pos, neg)
        self.ce_weight = torch.tensor(
            [(pos + neg) / neg, (pos + neg) / pos], dtype=torch.float32
        )

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]

        print(f'Loading Video: {video.pk} ...')

        summary = torch.LongTensor(flatten(self.summaries[idx]))
        scores = torch.FloatTensor(flatten(self.scores[idx])) / 0.1

        frames = flatten([story.frames for story in video.stories])
        frames = read_images(frames[::2])

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

        n_segments = int(self.num_frames / self.num_frames_per_segment)

        start_frame = 0

        summary = summary[start_frame: start_frame + n_segments]
        scores = scores[start_frame: start_frame + n_segments]

        if summary.shape[0] < n_segments:
            summary = torch.cat(
                (summary, torch.zeros((n_segments - len(summary)), dtype=torch.long))
            )
            scores = torch.cat(
                (scores, torch.zeros((n_segments - len(scores))))
            )

        summary = torch.LongTensor(summary)
        assert summary.shape[0] == n_segments
        assert scores.shape[0] == n_segments

        return {"video": video, "summary": summary, "scores": scores}
