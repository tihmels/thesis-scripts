#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
from pathlib import Path

import ffmpeg

from common.VideoData import get_frame_dir, get_date_time, get_frame_paths, VideoData
from common.fs_utils import re_create_dir, filename_match

parser = argparse.ArgumentParser()
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help="TS video files ")
parser.add_argument('--overwrite', action='store_false', dest='skip_existing',
                    help='Re-extracts frames for all videos')
parser.add_argument('--fps', type=float, default=0.0, help="extract frames per second")
parser.add_argument('--size', type=lambda s: list(map(int, s.split('x'))))


def extract_frames(vd: VideoData, fps=0.0, resize=None):
    stream = ffmpeg.input(vd.path)

    if fps > 0:
        stream = stream.filter('fps', fps=fps, round='up')

    if resize:
        stream = stream.filter('scale', resize[0], resize[1])

    stream = stream.output(f'{vd.frame_dir}/frame_%05d.jpg', **{'qscale:v': 1, 'qmin': 1, 'qmax': 1})
    ffmpeg.run(stream, quiet=True)


def was_processed(path: Path):
    frame_dir = get_frame_dir(path)

    if frame_dir.is_dir() and len(get_frame_paths(path)) > 0:
        return True

    return False


def main(args):
    video_files = {file for file in args.files if filename_match(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0

    video_files = sorted(video_files, key=get_date_time)

    print(f'Decoding {len(video_files)} videos', end='\n\n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vd}', end=' | ')

        re_create_dir(vd.frame_dir)

        extract_frames(vd, args.fps, args.size)

        print(f'{str(len(get_frame_paths(vd)))} Frames 'u'\u2713')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
