#!/Users/tihmels/Scripts/thesis/conda-env/bin/python

import argparse
import glob
import os
import shutil
from pathlib import Path

import ffmpeg


def extract_frames_from_video(video_file: Path, output_dir: Path, fps=0.0):
    stream = ffmpeg.input(video_file.absolute())

    if fps > 0:
        stream = stream.filter('fps', fps=fps)

    stream = stream.output(f'{output_dir}/frame_%05d.jpg')
    stream.run()


def extract_frames_from_video_files(video_files, fps=0.0, overwrite=False):
    videos = list(map(lambda v: Path(v), video_files))

    videos = [video for video in videos if video.is_file() and video.name.endswith('.mp4')]

    for video in videos:
        parent_dir = video.parent
        filename = video.name
        timecode = filename.split("-")[2]

        output_dir = Path(parent_dir, timecode)

        if output_dir.exists() and overwrite:
            shutil.rmtree(output_dir)
        elif output_dir.exists() and not overwrite:
            print(f'skipping {output_dir} because it already exists')
            continue

        output_dir.mkdir()

        extract_frames_from_video(video, output_dir, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('files', type=str, nargs='+')
    parser.add_argument('--fps', type=float, default=0.0)
    parser.add_argument('-r', '--recursive', action='store_true')
    parser.add_argument('-o', '--overwrite', action='store_true')

    args = parser.parse_args()

    files = []

    for file in args.files:
        if os.path.isfile(file) and file.endswith('.mp4'):
            files.append(file)
        elif os.path.isdir(file) and not args.recursive:
            [files.append(f) for f in glob.glob(f'{file}/*.mp4')]
        elif os.path.isdir(file) and args.recursive:
            [files.append(f) for f in glob.glob(f'{file}/**/*.mp4')]

    extract_frames_from_video_files(files, args.fps, args.overwrite)
