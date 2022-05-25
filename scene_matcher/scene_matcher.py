#!/Users/tihmels/Scripts/thesis-scripts/conda-env/bin/python

import argparse
import itertools
import re
from datetime import datetime
from enum import Enum
from functools import lru_cache
from itertools import groupby
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

TV_FILENAME_RE = r'TV-(\d{8})-(\d{4})-(\d{4}).webs.h264.mp4'


class VideoData:
    def __init__(self, path: Path):
        self._dateformat = '%Y%m%d'

        self.id: str = path.stem
        self.path: Path = path
        self.date: datetime = datetime.strptime(path.parent.name, self._dateformat)
        self.frame_dir: Path = Path(path.parent, path.name.split('-')[2])
        self.frames: [Path] = sorted(self.frame_dir.glob('frame_*.jpg'))
        self.n_frames: int = len(self.frames)
        self.segments: [(int, int)] = self.read_segments_from_file(Path(self.frame_dir, 'shots.txt'))
        self.n_segments: int = len(self.segments)

    @staticmethod
    def read_segments_from_file(file):
        shots = []

        file = open(file, 'r')
        for line in file.readlines():
            first_index, last_index = [int(x.strip(' ')) for x in line.split(' ')]
            shots.append((first_index, last_index))

        return shots

    def __str__(self):
        return "[" + str(self.__dict__['date'].strftime('%Y%m%d')) + "] " + str(
            self.__dict__['path'].id) + ": " + str(
            self.__dict__['n_frames']) + " frames, " + str(len(self.__dict__['segments'])) + " segments"


class VideoType(Enum):
    SUM = 0
    FULL = 1


class VideoStats:
    def __init__(self, vd: VideoData, vt: VideoType, segment_vector: np.array):
        self.id = vd.id
        self.date = vd.date.strftime("%Y%m%d")
        self.type = vt.name

        data = np.array([*vd.segments], dtype=np.int32)
        data = np.column_stack((data, np.array([s2 - s1 for s1, s2 in vd.segments], dtype=np.int32)))
        data = np.column_stack((data, segment_vector.astype(np.int32)))

        segment_center_indices = [np.round((s1 + s2) / 2).astype(int) for s1, s2 in vd.segments]
        representative_frames = np.array(vd.frames)[segment_center_indices]

        data = np.column_stack((data, representative_frames))

        self.df = pd.DataFrame(data=data, columns=['seg_start', 'seg_end', 'n_frames', 'match', 'rframe'])
        self.df.index += 1

    @property
    def n_segments(self):
        return self.df.shape[0]

    @property
    def n_frames(self):
        return sum(self.n_frames_per_segment)

    @property
    def n_frames_per_segment(self):
        return self.df.iloc[:, 2]

    @property
    def matched_segments(self):
        return self.df.iloc[:, 3]

    @property
    def n_segments_reused(self):
        return len(np.flatnonzero(self.matched_segments))

    @property
    def n_frames_reused(self):
        return sum(self.n_frames_per_segment[np.flatnonzero(self.matched_segments)])

    @property
    def segments_reused_ratio(self):
        return self.n_segments_reused / self.n_segments

    @property
    def frames_reused_ratio(self):
        return self.n_frames_reused / self.n_frames

    def print(self):
        n_segments = self.n_segments
        n_reused_segments = self.n_segments_reused
        total_frames = self.n_frames
        n_reused_frames = self.n_frames_reused

        print(f'Statistics for {self.id} ({self.type})')
        print("-------------------------------------------------")
        print(
            f'{n_reused_segments}/{n_segments} reused segments ({round(self.segments_reused_ratio * 100, 2)} %)')
        print(
            f'{n_reused_frames}/{total_frames} reused frames ({round(self.frames_reused_ratio * 100, 2)} %, {n_reused_frames / 25} sec)')
        print("-------------------------------------------------\n")

    def save_as_csv(self, dir_path: Path, suffix=""):
        if suffix:
            suffix = f'-{suffix}'

        self.df.to_csv(Path(dir_path, self.id + "-" + self.type + suffix + ".csv"))


def frame_similarity_detection(frame1: Image, frame2: Image, cutoff):
    hash1 = imagehash.dhash(frame1)
    hash2 = imagehash.dhash(frame2)

    return hash1 - hash2 < cutoff


def compare_framesets(frames1: [Image], frames2: [Image], cutoff=10):
    for f1, f2 in itertools.product(frames1, frames2):
        if frame_similarity_detection(f1, f2, cutoff):
            return True
    return False


@lru_cache(maxsize=128)
def get_image(path: str):
    return Image.open(path).convert('RGB')


def process_videos(date: str, videos: [VideoData], to_csv=False):
    if len(videos) < 2:
        print(f'not enough video data available for {date}')
        return

    main_video = max(videos, key=lambda v: v.n_frames)
    summary_videos = list(filter(lambda v: v is not main_video, videos))

    main_binary_segment_vector = np.zeros(main_video.n_segments)
    sum_binary_segment_dict = {summary.id: np.zeros(summary.n_segments) for summary in summary_videos}

    cutoff = 10

    print(
        f'Comparing {len(main_video.segments)} segments of {main_video.id}'
        f' with segments of {len(summary_videos)} summary videos ... \n')

    for seg_idx, (seg_start_idx, seg_end_idx) in enumerate(main_video.segments):
        print(
            f'[S{seg_idx + 1}/{main_video.n_segments}]: {seg_start_idx} - {seg_end_idx} ({seg_end_idx - seg_start_idx} frames)')

        main_frame_indices = np.round(np.linspace(seg_start_idx, seg_end_idx, 5)).astype(int)
        main_segment_frames = [Image.open(frame).convert('RGB') for frame in
                               np.array(main_video.frames)[main_frame_indices]]

        for summary in summary_videos:
            for sum_seg_idx, (sum_seg_start_idx, sum_seg_end_idx) in enumerate(summary.segments):

                sum_frame_indices = np.round(np.linspace(sum_seg_start_idx, sum_seg_end_idx, 3)).astype(int)
                sum_segment_frames = [get_image(str(frame)) for frame in np.array(summary.frames)[sum_frame_indices]]

                if compare_framesets(main_segment_frames, sum_segment_frames, cutoff):
                    main_binary_segment_vector[seg_idx] += 1
                    sum_binary_segment_dict[summary.id][sum_seg_idx] = seg_idx + 1

    main_video_stats = VideoStats(main_video, VideoType.FULL, main_binary_segment_vector)
    summary_video_stats = [VideoStats(video, VideoType.SUM, sum_binary_segment_dict[video.id]) for video in
                           summary_videos]

    print()
    main_video_stats.print()
    [s.print() for s in summary_video_stats]

    if to_csv:
        csv_dir = main_video.path.parent
        main_video_stats.save_as_csv(csv_dir, "co" + str(cutoff))
        [summary.save_as_csv(csv_dir, "co" + str(cutoff)) for summary in summary_video_stats]

    get_image.cache_clear()

    return main_video_stats, summary_video_stats


def check_requirements(video: Path):
    match = re.match(TV_FILENAME_RE, video.name)

    if match is None or not video.exists():
        print(f'{video.name} does not exist or does not match TV-*.mp4 pattern.')
        return False

    frame_dir = Path(video.parent, match.group(2))

    if not frame_dir.exists() or len(list(frame_dir.glob('frame_*.jpg'))) == 0:
        print(f'{video.name} no frames have been extracted.')
        return False

    if not Path(frame_dir, 'shots.txt').exists():
        print(f'{video.name} has no detected shots.')
        return False

    return True


def eval_video_statistics(vs: [VideoStats]):
    assert len(set([stats.type for stats in vs])) == 1

    print(f'Statistics of {len(vs)} videos ({vs[0].type})')
    print("-------------------------------------------------")

    total_segments = sum([stats.n_segments for stats in vs])
    total_segments_reused = sum([stats.n_segments_reused for stats in vs])

    segments_reused_ratio = total_segments_reused / total_segments
    segments_reused_perc = np.round(segments_reused_ratio * 100, 2)

    total_frames = sum([stats.n_frames for stats in vs])
    total_frames_reused = sum([stats.n_frames_reused for stats in vs])

    frames_reused_ratio = total_frames_reused / total_frames
    frames_reused_perc = np.round(frames_reused_ratio * 100, 2)

    print(f'Avg reused segments: {segments_reused_perc} %')
    print(f'Avg reused frames: {frames_reused_perc} %')
    print(f'Avg reused seconds: {total_frames_reused / len(vs) / 25}')
    print("-------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('dir', type=lambda p: Path(p).resolve(strict=True))
    parser.add_argument('--csv', action='store_true')
    args = parser.parse_args()

    tv_files = [file for file in args.dir.rglob('*.mp4') if check_requirements(file)]
    assert len(tv_files) > 0, "No TV-*.mp4 files could be found in " + str(args.dir)

    videos_by_date = dict([(date, list(videos)) for date, videos in groupby(tv_files, lambda f: f.name.split('-')[1])])

    main_video_statistics = []
    summary_video_statistics = []

    for date, files in videos_by_date.items():
        videos = [VideoData(file) for file in files]
        main_stats, sum_stats = process_videos(date, videos, args.csv)

        main_video_statistics.append(main_stats)
        summary_video_statistics.extend(sum_stats)

    eval_video_statistics(main_video_statistics)
    eval_video_statistics(summary_video_statistics)
