#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python

import argparse
import multiprocessing as mp
import os
import re
import shutil
from pathlib import Path

import ffmpeg

from utils.constants import TV_FILENAME_RE
from utils.fs_utils import get_frame_dir


def extract_frames_from_video(video: Path, fps=0.0, resize=(224, 224), overwrite=False,
                              prune=False):
    frame_dir = get_frame_dir(video)

    if prune and frame_dir.exists():
        shutil.rmtree(frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)

    stream = ffmpeg.input(video.absolute())

    if fps > 0:
        stream = stream.filter('fps', fps=fps, round='up')

    if resize:
        stream = stream.filter('scale', resize[0], resize[1])

    stream = stream.output(f'{frame_dir}/frame_%05d.jpg')
    ffmpeg.run(stream, overwrite_output=overwrite, quiet=True)

    return video


def check_requirements(path: Path, skip_existing):
    match = re.match(TV_FILENAME_RE, path.name)

    if match is None or not path.is_file():
        return False

    if skip_existing:
        frame_dir = get_frame_dir(path)

        if frame_dir.is_dir() and len(list(frame_dir.glob("frame_*.jpg"))) > 0:
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+',
                        help="video files or directories containing video files for frame extraction ")
    parser.add_argument('--fps', type=float, default=0.0, help="extract frames per second")
    parser.add_argument('-o', '--overwrite', action='store_true', help="overwrite existing frame files")
    parser.add_argument('-p', '--prune', action='store_true', help="prune all frame files if output directory exists")
    parser.add_argument('-s', '--skip', action='store_true', help="skip frame extraction if already exist")
    parser.add_argument('--size', type=lambda s: list(map(int, s.split('x'))))
    parser.add_argument('--parallel', action='store_true',
                        help="execute frame extraction using parallel multiprocessing")
    args = parser.parse_args()

    videos = []

    for file in args.files:
        if file.is_file() and check_requirements(file, args.skip):
            videos.append(file)
        elif file.is_dir():
            [videos.append(f) for f in sorted(file.glob('*.mp4')) if check_requirements(f, args.skip)]

    assert len(videos) != 0

    print(f'Frame extraction for {len(videos)} videos ...')


    def callback_handler(res):
        if res is not None and isinstance(res, Path):
            print(f'{res.name} done')


    if args.parallel:
        with mp.Pool(os.cpu_count()) as pool:
            [pool.apply_async(extract_frames_from_video, (video,),
                              kwds={'fps': args.fps, 'overwrite': args.overwrite,
                                    'prune': args.prune, 'resize': args.size},
                              callback=callback_handler) for video in videos]
            pool.close()
            pool.join()

    else:
        for video in videos:
            result = extract_frames_from_video(video, args.fps, args.size, args.overwrite, args.prune)
            callback_handler(result)
