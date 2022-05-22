#!/Users/tihmels/Scripts/thesis/conda-env/bin/python

import argparse
import multiprocessing as mp
import os
import re
import shutil
from pathlib import Path

import ffmpeg

TV_FILENAME_RE = r'TV-(\d{8})-(\d{4})-(\d{4}).webs.h264.mp4'


def extract_frames_from_video(video: Path, fps=0.0, overwrite=False, prune=False):
    match = re.match(TV_FILENAME_RE, video.name)

    if match is None:
        print(f'{video} does not match TV pattern. Skip ...')
        return

    output_dir = Path(video.parent, match.group(2))

    if prune and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(exist_ok=True)

    stream = ffmpeg.input(video.absolute())

    if fps > 0:
        stream = stream.filter('fps', fps=fps)

    stream = stream.output(f'{output_dir}/frame_%05d.jpg')
    ffmpeg.run(stream, overwrite_output=overwrite, quiet=True)

    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+',
                        help="video files or directories containing video files for frame extraction ")
    parser.add_argument('--fps', type=float, default=0.0, help="extract frames per second")
    parser.add_argument('-r', '--recursive', action='store_true', help="search recursively for video files")
    parser.add_argument('-o', '--overwrite', action='store_true', help="overwrite existing frame files")
    parser.add_argument('-p', '--prune', action='store_true', help="prune all frame files if output directory exists")
    parser.add_argument('--parallel', action='store_true',
                        help="execute frame extraction using parallel multiprocessing")
    args = parser.parse_args()

    tv_files = []

    for file in args.files:
        if file.is_file() and re.match(TV_FILENAME_RE, file.name):
            tv_files.append(file)
        elif file.is_dir() and not args.recursive:
            [tv_files.append(f) for f in file.glob('*.mp4') if re.match(TV_FILENAME_RE, f.name)]
        elif file.is_dir() and args.recursive:
            [tv_files.append(f) for f in file.rglob('*.mp4') if re.match(TV_FILENAME_RE, f.name)]

    if args.parallel:
        with mp.Pool(os.cpu_count()) as pool:
            [pool.apply_async(extract_frames_from_video, (file,),
                              kwds={'fps': args.fps, 'overwrite': args.overwrite, 'prune': args.prune},
                              callback=lambda f: print(f'{f.name} done')) for file in tv_files]
            pool.close()
            pool.join()

    else:
        for file in tv_files:
            extract_frames_from_video(file, args.fps, args.overwrite, args.prune)
            print(f'{file.name} done')
