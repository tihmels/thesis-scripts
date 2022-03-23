#!/Users/tihmels/Scripts/thesis/conda-env/bin/python

import argparse
from itertools import groupby, chain
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from VideoData import VideoFile


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
                print(f'[{main_video.timecode}] Scene {index}, {scene.first_frame_path} - {sum_scene.first_frame_path}')
                binary_scene_vector[index] = 1
                break

            if frame_similarity_detection(last_frame, sum_last_frame):
                print(f'[{main_video.timecode}] Scene {index}, {scene.last_frame_path} - {sum_scene.last_frame_path}')
                binary_scene_vector[index] = 1
                break

    print(f'{np.count_nonzero(binary_scene_vector == 1)} matched scenes detected')
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

        binary_scene_vector = match_scenes(main_broadcast, summary_broadcasts)

        return binary_scene_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=lambda p: Path(p).resolve(strict=True))
    args = parser.parse_args()

    mp4_files = [VideoFile(path) for path in list(args.dir.rglob('*.mp4'))]
    assert len(mp4_files) > 0, "no .mp4 files present in " + str(args.dir)

    mp4_files = list(filter(lambda video: video.check_requirements(), mp4_files))
    for video in mp4_files:
        video.load_scene_data()

    videos_by_date = dict()
    for date, videos in groupby(mp4_files, lambda vd: str(vd.timecode)):
        videos_by_date[date] = list(videos)

    process_videos_by_date(videos_by_date)
