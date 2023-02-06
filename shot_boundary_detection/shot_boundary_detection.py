#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common.VAO import VAO, get_shot_file, get_data_dir, get_frame_dir, get_frame_paths, get_date_time, \
    get_main_transcript_file, read_shots_from_file
from common.constants import TV_FILENAME_RE
from common.utils import read_images
from post_sbd import fix_first_anchorshot_segment
from transnetv2 import TransNetV2

parser = argparse.ArgumentParser('Shot Boundary Detection (SBD) using TransNet V2')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help="Tagesschau video file(s)")
parser.add_argument('--threshold', type=float, default=0.2, help='Shot transition threshold')
parser.add_argument('--img', action='store_true')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing',
                    help="Re-calculate shot boundaries for all videos")

model = TransNetV2()


def shot_transition_detection(frames, threshold=0.2):
    single_frame_predictions, all_frame_predictions = model.predict_frames(frames)

    shots = model.predictions_to_scenes(single_frame_predictions, threshold=threshold)
    img = model.visualize_predictions(frames, (single_frame_predictions, all_frame_predictions))

    return shots, img


def process_video(vao: VAO, threshold):
    frames = read_images(vao.data.frames, resize=(48, 27))

    shots, img = shot_transition_detection(np.array(frames), threshold)

    if not vao.is_summary and get_main_transcript_file(vao).is_file():
        transcript = vao.data.transcripts
        shots = fix_first_anchorshot_segment(shots, transcript)

    shots = shots[shots[:, 1] - shots[:, 0] > 12]

    return shots, img


def check_requirements(video: Path):
    if not re.match(TV_FILENAME_RE, video.name):
        return False

    frame_path = get_frame_dir(video)

    if not frame_path.is_dir() or not len(get_frame_paths(video)) > 0:
        print(f'{video.name} has no extracted frames.')
        return False

    return True


def was_processed(video: Path):
    shot_file = get_shot_file(video)

    if not shot_file.is_file():
        return False

    frames = get_frame_paths(video)

    shots = read_shots_from_file(shot_file)
    return len(frames) - 15 < shots[-1].last_frame_idx < len(frames)


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    print(f'Detecting shot boundaries for {len(video_files)} videos ...', end='\n\n')

    for idx, vf in enumerate(video_files):
        vao = VAO(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vao}')

        shots, img = process_video(vao, args.threshold)

        df = pd.DataFrame(data=shots, columns=['first_frame_idx', 'last_frame_idx'])

        df.to_csv(get_shot_file(vao), index=False)

        if args.img:
            img.save(Path(get_data_dir(vao), 'shots.png').absolute())


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
