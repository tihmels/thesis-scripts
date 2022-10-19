#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import re
from pathlib import Path

import ffmpeg

from common.VideoData import get_frame_dir, get_date_time, get_frame_paths, VideoData
from common.constants import TV_FILENAME_RE
from common.fs_utils import re_create_dir

parser = argparse.ArgumentParser('Video frame extraction using ffmpeg')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help='Tagesschau video file(s)')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing',
                    help='Re-extract frames for all videos')
parser.add_argument('--fps', type=float, default=0.0, help="Frames per second to extract")
parser.add_argument('--size', type=lambda s: list(map(int, s.split('x'))), help="Scale frames to size (e.g. 224x224)")


def extract_frames(vd: VideoData, fps=0.0, resize=None):
    stream = ffmpeg.input(vd.path)

    if fps > 0:
        stream = stream.filter('fps', fps=fps, round='up')

    if resize:
        stream = stream.filter('scale', resize[0], resize[1])

    stream = stream.output(f'{vd.frame_dir}/frame_%05d.jpg', **{'qscale:v': 1, 'qmin': 1, 'qmax': 1})
    ffmpeg.run(stream, quiet=True)


def was_processed(video: Path):
    frame_dir = get_frame_dir(video)

    if frame_dir.is_dir() and len(get_frame_paths(video)) > 0:
        print(f'{video.name} has already frames extracted.')
        return True

    return False


def main(args):
    video_files = {file for file in args.files if re.match(TV_FILENAME_RE, file.name)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    print(f'Extracting frames from {len(video_files)} videos ...', end='\n\n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vd}', end=' | ')

        re_create_dir(vd.frame_dir)

        extract_frames(vd, args.fps, args.size)

        print(f'{str(len(get_frame_paths(vd)))} Frames 'u'\u2713')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
