#!/Users/tihmels/Scripts/thesis/conda-env/bin/python

import argparse
import re
from itertools import groupby, chain
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from VideoData import VideoFile

TV_FILENAME_RE = r'TV-(\d{8})-(\d{4})-(\d{4}).webs.h264.mp4'


class VideoData:
    def __init__(self, path: Path):
        self.path: Path = path
        self.frame_dir: Path = Path(path.parent, path.name.split('-')[2])
        self.frames: [Path] = sorted(self.frame_dir.glob('frame_*.jpg'))
        self.n_frames: int = len(self.frames)
        self.segments: [(int, int)] = self.read_segments_from_file(Path(self.frame_dir, 'shots.txt'))

    @staticmethod
    def read_segments_from_file(file):
        shots = []

        file = open(file, 'r')
        for line in file.readlines():
            first_index, last_index = [int(x.strip(' ')) for x in line.split(' ')]
            shots.append((first_index, last_index))

        return shots

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


def frame_similarity_detection(frame1: Image, frame2: Image, cutoff=12):
    hash1 = imagehash.average_hash(frame1)
    hash2 = imagehash.average_hash(frame2)

    return hash1 - hash2 < cutoff


def match_scenes(main_video: VideoFile, summary_videos):
    binary_scene_vector = np.zeros(main_video.n_scenes)
    summary_scenes = list(flatmap(lambda v: v.scenes, summary_videos))

    for index, scene in enumerate(main_video.scenes):

        first_frame, last_frame = scene.load_scene_frames()

        for sum_scene in summary_scenes:

            sum_first_frame, sum_last_frame = sum_scene.load_scene_frames()

            if frame_similarity_detection(first_frame, sum_first_frame):
                print(f'[{main_video.date}] Scene {index}, {scene.first_frame_path} - {sum_scene.first_frame_path}')
                binary_scene_vector[index] = 1
                break

            if frame_similarity_detection(last_frame, sum_last_frame):
                print(f'[{main_video.date}] Scene {index}, {scene.last_frame_path} - {sum_scene.last_frame_path}')
                binary_scene_vector[index] = 1
                break

    print(f'{np.count_nonzero(binary_scene_vector == 1)}/{len(binary_scene_vector)} matched scenes detected')

    return binary_scene_vector


def flatmap(func, *iterable):
    return chain.from_iterable(map(func, *iterable))


def process_videos_by_date(videos_by_date):
    for timecode, videos in videos_by_date.items():

        if len(videos) < 2:
            print(f'not enough video data available for {timecode}')
            continue

        main_broadcast = max(videos, key=lambda v: v.n_frames)
        summary_broadcasts = list(filter(lambda v: v is not main_broadcast, videos))

        print(
            f'[{timecode}] Scene Matching ({main_broadcast.name} : {", ".join(list(map(lambda v: v.name, summary_broadcasts)))})')

        binary_scene_vector = np.array(match_scenes(main_broadcast, summary_broadcasts), dtype=np.int64)

        scenes = np.array(main_broadcast.load_scenes_from_file())

        print(f'scenes shape {scenes.shape}')
        print(scenes.dtype)
        print(scenes[:10, :])

        binary_scene_vector = binary_scene_vector[np.newaxis].T

        print(f'binary scene vector {binary_scene_vector.shape}')
        print(binary_scene_vector[:10])

        binary_scene_vector = np.append(scenes, binary_scene_vector, axis=1)

        print(binary_scene_vector)

        return binary_scene_vector


def process_videos(date: str, videos: [VideoData]):

    if len(videos) < 2:
        print(f'not enough video data available for {date}')
        return

    main_video = max(videos, key=lambda v: v.n_frames)
    summary_videos = list(filter(lambda v: v is not main_video, videos))

    for segment_start_idx, segment_end_idx in main_video.segments:
        print(f'{segment_start_idx} - {segment_end_idx}')
        segment_frames = [Image.open(f).convert('L') for f in main_video.frames[segment_start_idx:segment_end_idx]]

        for summary in summary_videos:
            for sum_segment_start_idx, sum_segment_end_idx in summary.segments:
                sum_segment_frames = [Image.open(f).convert('L') for f in
                                      summary.frames[sum_segment_start_idx:sum_segment_end_idx]]

                if frame_similarity_detection(segment_frames[0], sum_segment_frames[0]):
                    print('Match')
                    break


def check_requirements(video: Path):
    match = re.match(TV_FILENAME_RE, video.name)

    if match is None or not video.exists():
        return False

    frame_dir = Path(video.parent, match.group(2))

    if not frame_dir.exists() or len(list(frame_dir.glob('frame_*.jpg'))) == 0:
        return False

    if not Path(frame_dir, 'shots.txt').exists():
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('dir', type=lambda p: Path(p).resolve(strict=True))
    args = parser.parse_args()

    tv_files = [file for file in args.dir.rglob('*.mp4') if check_requirements(file)]
    assert len(tv_files) > 0, "No TV-*.mp4 files could be found in " + str(args.dir)

    videos_by_date = dict([(date, list(videos)) for date, videos in groupby(tv_files, lambda f: f.name.split('-')[1])])

    for date, files in videos_by_date.items():
        videos = [VideoData(file) for file in files]
        process_videos(date, videos)
