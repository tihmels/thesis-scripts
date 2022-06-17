#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import itertools
import re
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from PIL.Image import Resampling

from VideoData import VideoData, VideoType, VideoStats, get_vs_evaluation_df
from utils.constants import TV_FILENAME_RE, MAIN_VIDEOS_PATH
from utils.fs_utils import get_frame_dir, get_shot_file, get_date_time, get_summary_videos, get_data_dir


def frame_hash_distance(f1: Image, f2: Image):
    hash1 = imagehash.dhash(f1)
    hash2 = imagehash.dhash(f2)

    return hash1 - hash2


def min_frameset_hash_distance(main_frames: [Image], sum_frames: [Image]):
    return min([frame_hash_distance(f1, f2) for f1, f2 in itertools.product(main_frames, sum_frames)])


@lru_cache(maxsize=1024)
def get_image_cached(file):
    return get_image(file)


def get_image(file):
    return Image.open(str(file)).convert('L').resize((9, 8), Resampling.LANCZOS)


def was_processed(video: VideoData):
    date, timecode = video.date_str, video.timecode
    pattern = re.compile(r"^TV-" + re.escape(date) + r"-" + re.escape(timecode) + r"-\S*.csv$")

    for file in video.path.parent.iterdir():
        if file.is_file() and pattern.match(file.name):
            return True

    return False


def process_videos(date: str, main_video: VideoData, summary_videos: [VideoData], cutoff: int, skip_existing=False,
                   to_csv=False):
    if len(summary_videos) < 2:
        print(f'Not enough video data available for {date}')
        return

    # TODO: fix was_processed
    if skip_existing and all(was_processed(video) for video in videos):
        print(f'All {len(videos)} videos for {date} already processed. Skip ... ')
        return

    main_segment_vector = np.zeros(main_video.n_segments)
    sum_segment_dict = {summary.id: (np.zeros(summary.n_segments), np.full(summary.n_segments, np.inf)) for summary in
                        summary_videos}

    print(
        f'Comparing {len(main_video.segments)} segments of {main_video.id}'
        f' with segments of {len(summary_videos)} summary videos \n')

    for seg_idx, (seg_start_idx, seg_end_idx) in enumerate(main_video.segments):

        print('{:<45s}'.format(
            "[S{}/{}]: F{} - F{} ({} frames)".format(seg_idx + 1, main_video.n_segments, seg_start_idx, seg_end_idx,
                                                     seg_end_idx - seg_start_idx + 1)), end="")

        main_frame_indices = np.round(np.linspace(seg_start_idx + 5, seg_end_idx - 5, 5)).astype(int)
        main_segment_frames = [get_image(frame) for frame in
                               np.array(main_video.frames)[main_frame_indices]]

        for summary in summary_videos:

            segment_distances = np.full(summary.n_segments, np.inf)

            for sum_seg_idx, (sum_seg_start_idx, sum_seg_end_idx) in enumerate(summary.segments):
                sum_frame_indices = np.round(np.linspace(sum_seg_start_idx + 5, sum_seg_end_idx - 5, 3)).astype(int)
                sum_segment_frames = [get_image_cached(frame) for frame in np.array(summary.frames)[sum_frame_indices]]

                min_frame_dist = min_frameset_hash_distance(main_segment_frames, sum_segment_frames)
                segment_distances[sum_seg_idx] = min_frame_dist

            if any(segment_distances < cutoff):
                main_segment_vector[seg_idx] += 1

                seg_min_dist_idx = np.argmin(segment_distances)
                seg_min_dist = segment_distances[seg_min_dist_idx]

                sum_seg_matched, sum_seg_dist = sum_segment_dict[summary.id]

                if sum_seg_matched[seg_min_dist_idx] == 0 or sum_seg_dist[seg_min_dist_idx] > seg_min_dist:
                    sum_seg_matched[seg_min_dist_idx] = seg_idx + 1
                    sum_seg_dist[seg_min_dist_idx] = seg_min_dist

                sum_segment_dict[summary.id] = (sum_seg_matched, sum_seg_dist)

                print("x", end="")
            else:
                print(".", end="")

        print()

        [f.close() for f in main_segment_frames]

    main_video_stats = VideoStats(main_video, VideoType.FULL, main_segment_vector)
    summary_video_stats = [VideoStats(summary_video, VideoType.SUM, sum_segment_dict[summary_video.id]) for
                           summary_video in summary_videos]

    print()
    main_video_stats.print()
    [s.print() for s in summary_video_stats]

    if to_csv:
        csv_dir = Path(get_data_dir(main_video.path), "sm")
        main_video_stats.save_as_csv(csv_dir, "co" + str(cutoff))
        [summary.save_as_csv(csv_dir, "co" + str(cutoff)) for summary in summary_video_stats]

    print(get_image_cached.cache_info())
    get_image_cached.cache_clear()

    return main_video_stats, summary_video_stats


def check_requirements(video: Path):
    match = re.match(TV_FILENAME_RE, video.name)

    if match is None or not video.is_file():
        print(f'{video.name} does not exist or does not match TV-*.mp4 pattern.')
        return False

    frame_dir = get_frame_dir(video)

    if not frame_dir.is_dir() or len(list(frame_dir.glob('frame_*.jpg'))) == 0:
        print(f'{video.name} no frames have been extracted.')
        return False

    if not get_shot_file(video).exists():
        print(f'{video.name} has no detected shots.')
        return False

    return True


def get_binary_segment_vector(vs: VideoStats):
    return np.where(vs.matched_segments == 0, 0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
    parser.add_argument('-s', '--skip', action='store_true', help="skip scene matching if already exist")
    parser.add_argument('--cutoff', type=int, choices=range(1, 30), default=10, help="set hash similarity cutoff")
    parser.add_argument('--csv', action='store_true', help="store scene matching results in a csv file")
    args = parser.parse_args()

    video_files = []

    for file in args.files:
        if file.is_file() and check_requirements(file):
            video_files.append(file)
        elif file.is_dir():
            video_files.extend([video for video in sorted(file.glob('*.mp4')) if check_requirements(video)])

    assert len(video_files) > 0, "No TV-*.mp4 files could be found in " + str(args.dirs)

    video_files.sort(key=get_date_time)

    videos_by_date = {get_date_time(video): video for video in video_files}

    summary_videos = [video for video in get_summary_videos() if check_requirements(video)]
    summaries_by_date = {get_date_time(video): video for video in summary_videos}

    for idx, (date, video) in enumerate(videos_by_date.items()):

        main_video = VideoData(video)

        (rangeStart, rangeEnd) = date - timedelta(hours=8), date + timedelta(hours=16)
        summary_videos = [VideoData(video) for date, video in summaries_by_date.items() if
                          rangeStart <= date <= rangeEnd]

        print(f'\n[{idx + 1}/{len(videos_by_date)}] {date}')
        result = process_videos(date, main_video, summary_videos, args.cutoff, args.skip, args.csv)

        if result:
            main_vs, sum_vs = result
            df = get_vs_evaluation_df([main_vs], sum_vs)
            df.index += idx

            bin_seg_vec = get_binary_segment_vector(main_vs)
            filename = f'{main_vs.id}-BINSEGVEC.txt'
            np.savetxt(str(Path(get_data_dir(video), filename)), bin_seg_vec, fmt='%i')

            output_file = Path(MAIN_VIDEOS_PATH, "statistics-co" + str(args.cutoff) + ".csv")
            df.to_csv(str(output_file), mode='a', header=not output_file.exists())
