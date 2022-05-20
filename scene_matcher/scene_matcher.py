#!/Users/tihmels/Scripts/thesis/conda-env/bin/python

import argparse
from itertools import groupby, chain
from pathlib import Path

import imagehash
import re
import os
import numpy as np
from PIL import Image

from VideoData import VideoFile

TV_FILENAME_RE = r'TV-(\d{8})-(\d{4})-(\d{4}).webs.h264.mp4'

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('dirs', type=lambda p: Path(p).resolve(strict=True))
    args = parser.parse_args()

    mp4_files = [VideoFile(file) for file in list(args.dir.rglob('*.mp4')) if re.match(TV_FILENAME_RE, os.path.basename(file))]
    assert len(mp4_files) > 0, "no TV-*.mp4 files could be found in " + str(args.dir)

    mp4_files = list(filter(lambda video: video.check_requirements(), mp4_files))
    for video in mp4_files:
        video.load_scene_data()

    videos_by_date = dict()
    for date, videos in groupby(mp4_files, lambda vd: str(vd.date)):
        videos_by_date[date] = list(videos)

    process_videos_by_date(videos_by_date)
