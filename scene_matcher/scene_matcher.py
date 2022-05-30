#!/Users/tihmels/Scripts/thesis-scripts/conda-env/bin/python -u

import argparse
import itertools
import re
from datetime import datetime
from functools import lru_cache
from itertools import groupby
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from VideoData import VideoData, VideoStats, VideoType, get_vs_eval_df

TV_FILENAME_RE = r'TV-(\d{8})-(\d{4})-(\d{4}).webs.h264.mp4'


def frame_similarity_detection(frame1: Image, frame2: Image, cutoff):
    hash1 = imagehash.dhash(frame1)
    hash2 = imagehash.dhash(frame2)

    return hash1 - hash2 < cutoff


def compare_framesets(frames1: [Image], frames2: [Image], cutoff=8):
    for f1, f2 in itertools.product(frames1, frames2):
        if frame_similarity_detection(f1, f2, cutoff):
            return True
    return False


@lru_cache(maxsize=1024)
def get_image(path: str):
    return Image.open(path).convert('L').resize((9, 8), Image.ANTIALIAS)


def was_processed(video: VideoData):
    date, timecode = video.date_str, video.timecode
    pattern = re.compile(r"^TV-" + re.escape(date) + r"-" + re.escape(timecode) + r"-\S*.csv$")

    for file in video.path.parent.iterdir():
        if file.is_file() and pattern.match(file.name):
            return True

    return False


def process_videos(date: str, videos: [VideoData], skip_existing=False, to_csv=False):
    if len(videos) < 2:
        print(f'Not enough video data available for {date}')
        return

    if skip_existing and all(was_processed(video) for video in videos):
        print(f'All {len(videos)} videos for {date} already processed. Skip ... ')
        return

    main_video = max(videos, key=lambda v: v.n_frames)
    summary_videos = sorted(list(filter(lambda v: v is not main_video, videos)), key=lambda s: s.date)

    main_segment_vector = np.zeros(main_video.n_segments)
    sum_segment_dict = {summary.id: np.zeros(summary.n_segments) for summary in summary_videos}

    cutoff = 12

    print(
        f'Comparing {len(main_video.segments)} segments of {main_video.id}'
        f' with segments of {len(summary_videos)} summary videos ... \n')

    for seg_idx, (seg_start_idx, seg_end_idx) in enumerate(main_video.segments):
        print(
            f'[S{seg_idx + 1}/{main_video.n_segments}]: F{seg_start_idx} - F{seg_end_idx} ({seg_end_idx - seg_start_idx + 1} frames)',
            end="\t")

        main_frame_indices = np.round(np.linspace(seg_start_idx, seg_end_idx, 5)).astype(int)
        main_segment_frames = [Image.open(frame).convert('L').resize((9, 8), Image.ANTIALIAS) for frame in
                               np.array(main_video.frames)[main_frame_indices]]

        for summary in summary_videos:
            match = False

            for sum_seg_idx, (sum_seg_start_idx, sum_seg_end_idx) in enumerate(summary.segments):

                sum_frame_indices = np.round(np.linspace(sum_seg_start_idx, sum_seg_end_idx, 3)).astype(int)
                sum_segment_frames = [get_image(str(frame)) for frame in np.array(summary.frames)[sum_frame_indices]]

                if compare_framesets(main_segment_frames, sum_segment_frames, cutoff):
                    main_segment_vector[seg_idx] += 1
                    match = True

                    if sum_segment_dict[summary.id][sum_seg_idx] == 0:
                        sum_segment_dict[summary.id][sum_seg_idx] = seg_idx + 1

                    break

            print("x", end="") if match else print(".", end="")
        print()

        [f.close() for f in main_segment_frames]

    main_video_stats = VideoStats(main_video, VideoType.FULL, main_segment_vector)
    summary_video_stats = [VideoStats(video, VideoType.SUM, sum_segment_dict[video.id]) for video in
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('dirs', type=lambda p: Path(p).resolve(strict=True), nargs='+')
    parser.add_argument('-r', '--recursive', action='store_true', help="search recursively for TV-*.mp4 files")
    parser.add_argument('-s', '--skip', action='store_true', help="skip scene matching if already exist")
    parser.add_argument('--csv', action='store_true', help="store scene matching results in a csv file")
    parser.add_argument('--append', action='store_true', help="append to csv file")
    args = parser.parse_args()

    tv_files = []

    if args.recursive:
        for directory in args.dirs:
            tv_files.extend([file for file in directory.rglob('*.mp4') if check_requirements(file)])
    else:
        for directory in args.dirs:
            tv_files.extend([file for file in directory.glob('*.mp4') if check_requirements(file)])

    assert len(tv_files) > 0, "No TV-*.mp4 files could be found in " + str(args.dirs)

    tv_files.sort(key=lambda file: datetime.strptime(file.name.split("-")[1] + file.name.split("-")[2], "%Y%m%d%H%M"))

    videos_by_date = dict([(date, list(videos)) for date, videos in groupby(tv_files, lambda f: f.parent.name)])

    main_video_statistics = []
    summary_video_statistics = []

    for idx, (date, files) in enumerate(videos_by_date.items()):
        videos = [VideoData(file) for file in files]

        print(f'\n[{idx + 1}/{len(videos_by_date)}] {date}')
        stats = process_videos(date, videos, args.skip, args.csv)

        if stats:
            main_stats, sum_stats = stats
            main_video_statistics.append(main_stats)
            summary_video_statistics.extend(sum_stats)

    df = get_vs_eval_df(main_video_statistics, summary_video_statistics)

    output_path = Path(Path.home(), "TV", "stats.csv")
    if args.append:
        df.to_csv(str(output_path), mode='a', header=not output_path.exists())
    else:
        df.to_csv(str(output_path))
