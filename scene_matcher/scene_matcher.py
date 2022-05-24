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
    FULL = 0
    SUMMARY = 1


class VideoStats:
    def __init__(self, vd: VideoData, vt: VideoType, segment_vector: np.array):
        self.name = vd.id
        self.date = vd.date.strftime("%d.%m.%Y")
        self.vt = vt.name
        self.segments = vd.segments
        self.seg_vec = segment_vector

        # matched_segments_indices = np.where(main_binary_segment_vector)
        # matched_segments_n_frames = [i2 - i1 for i1, i2 in np.array(main_video.segments)[matched_segments_indices]]


def frame_similarity_detection(frame1: Image, frame2: Image, cutoff):
    hash1 = imagehash.dhash(frame1)
    hash2 = imagehash.dhash(frame2)

    return hash1 - hash2 < cutoff


def compare_framesets(frames1: [Image], frames2: [Image], cutoff=8):
    for f1, f2 in itertools.product(frames1, frames2):
        if frame_similarity_detection(f1, f2, cutoff):
            return True
    return False


@lru_cache(maxsize=256)
def get_image(path: str):
    return Image.open(path).convert('RGB')


def process_videos(date: str, videos: [VideoData]):
    if len(videos) < 2:
        print(f'not enough video data available for {date}')
        return

    main_video = max(videos, key=lambda v: v.n_frames)
    summary_videos = list(filter(lambda v: v is not main_video, videos))

    main_binary_segment_vector = np.zeros(main_video.n_segments)
    sum_binary_segment_dict = {summary.id: np.zeros(summary.n_segments) for summary in summary_videos}

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

                if compare_framesets(main_segment_frames, sum_segment_frames):
                    main_binary_segment_vector[seg_idx] = 1
                    sum_binary_segment_dict[summary.id][sum_seg_idx] = seg_idx + 1
                    break

    # TODO: matched_segments_indices & n_reused_segments difference?
    matched_segments_indices = np.where(main_binary_segment_vector)
    matched_segments_n_frames = [i2 - i1 for i1, i2 in np.array(main_video.segments)[matched_segments_indices]]

    n_reused_segments = np.count_nonzero(main_binary_segment_vector == 1)
    n_reused_frames = sum(matched_segments_n_frames)

    print(
        f'\n{n_reused_segments}/{len(main_binary_segment_vector)} reused segments ({round((n_reused_segments / len(main_binary_segment_vector)) * 100, 2)} %)')
    print(
        f'{n_reused_frames}/{main_video.n_frames} reused frames ({round((n_reused_frames / main_video.n_frames) * 100, 2)} %, {n_reused_frames / 25} sec)\n')

    get_image.cache_clear()


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('dir', type=lambda p: Path(p).resolve(strict=True))
    args = parser.parse_args()

    tv_files = [file for file in args.dir.rglob('*.mp4') if check_requirements(file)]
    assert len(tv_files) > 0, "No TV-*.mp4 files could be found in " + str(args.dir)

    videos_by_date = dict([(date, list(videos)) for date, videos in groupby(tv_files, lambda f: f.id.split('-')[1])])

    for date, files in videos_by_date.items():
        videos = [VideoData(file) for file in files]
        process_videos(date, videos)
