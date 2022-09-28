#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import multiprocessing as mp
import os
import re
from pathlib import Path
from shutil import rmtree

import ffmpeg

from common.VideoData import get_frame_dir, get_date_time, get_frame_paths, VideoData
from common.constants import TV_FILENAME_RE


def extract_frames(vd: VideoData, fps=0.0, resize=None):
    frame_dir = vd.frame_dir

    if frame_dir.is_dir():
        rmtree(frame_dir)

    frame_dir.mkdir(parents=True, exist_ok=True)

    stream = ffmpeg.input(vd.path)

    if fps > 0:
        stream = stream.filter('fps', fps=fps, round='up')

    if resize:
        stream = stream.filter('scale', resize[0], resize[1])

    stream = stream.output(f'{frame_dir}/frame_%05d.jpg', **{'qscale:v': 1, 'qmin': 1, 'qmax': 1})
    ffmpeg.run(stream, quiet=True)

    return vd


def check_requirements(path: Path, skip_existing: bool):
    match = re.match(TV_FILENAME_RE, path.name)

    if match is None or not path.is_file():
        return False

    if skip_existing:
        frame_dir = get_frame_dir(path)

        if frame_dir.is_dir() and len(get_frame_paths(path)) > 0:
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+',
                        help="video files or directories containing video files for frame extraction ")
    parser.add_argument('--overwrite', action='store_true', help='Re-extracts frames for all videos')
    parser.add_argument('--fps', type=float, default=0.0, help="extract frames per second")
    parser.add_argument('--size', type=lambda s: list(map(int, s.split('x'))))
    parser.add_argument('--parallel', action='store_true',
                        help="execute frame extraction using parallel multiprocessing")
    args = parser.parse_args()

    video_files = []

    for file in args.files:
        if file.is_file() and check_requirements(file, not args.overwrite):
            video_files.append(file)
        elif file.is_dir():
            video_files.extend([f for f in file.glob('*.mp4') if check_requirements(f, not args.overwrite)])

    assert len(video_files) != 0

    video_files.sort(key=get_date_time)

    print(f'Decoding {len(video_files)} videos\n')


    def callback_handler(res):
        if res is not None and isinstance(res, VideoData):
            print(f'{res} done')
        else:
            print(f'There was a problem ...')


    if args.parallel:
        with mp.Pool(os.cpu_count()) as pool:
            [pool.apply_async(extract_frames, (VideoData(vf),),
                              kwds={'fps': args.fps, 'resize': args.size},
                              callback=callback_handler) for vf in video_files]
            pool.close()
            pool.join()

    else:
        for idx, vf in enumerate(video_files):
            vd = VideoData(vf)

            print(f'[{idx + 1}/{len(video_files)}] {vd}', end=' ... ')

            extract_frames(vd, args.fps, args.size)

            print('Done')
