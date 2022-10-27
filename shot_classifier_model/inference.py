#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import json
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorflow as tf

from common.VideoData import get_keyframe_dir, get_date_time, VideoData
from common.constants import TV_FILENAME_RE

model = tf.keras.models.load_model(Path(Path(__file__).resolve().parent, 'model', 'ts_anchor_model'))
model.load_weights(Path(Path(__file__).resolve().parent, 'model', 'ts_anchor_v1.weights.best.hdf5'))

input_shape = model.input_shape[1:-1]

parser = ArgumentParser('Anchorshot Detection')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
parser.add_argument('--topn', type=int, default=5)

with open(Path(Path(__file__).resolve().parent, 'model', 'classes.txt'), 'r') as file:
    classes = json.load(file)


def classify_video_shots(vd: VideoData, top_n=5):
    keyframes = [tf.keras.preprocessing.image.load_img(frame, target_size=input_shape) for frame in vd.keyframes]
    keyframes = [tf.keras.preprocessing.image.img_to_array(frame) for frame in keyframes]
    keyframes = [np.expand_dims(frame, axis=0) for frame in keyframes]

    for kf in keyframes:
        kf = tf.keras.applications.resnet50.preprocess_input(kf)
        prediction = model.predict(kf)[0]

        result = [(classes[i], np.round(float(prediction[i]) * 100.0, 2)) for i in range(len(prediction))]
        result.sort(reverse=True, key=lambda x: x[1])

        yield result[:top_n]


def check_requirements(path: Path):
    assert path.parent.name == 'ts15'

    match = re.match(TV_FILENAME_RE, path.name)

    if match is None or not path.is_file():
        print(f'{path.name} does not exist or does not match TV-*.mp4 pattern.')
        return False

    kf_dir = get_keyframe_dir(path)

    if not kf_dir.is_dir() or len(list(kf_dir.glob("*.jpg"))) < 1:
        return False

    return True


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    for vf in video_files:
        vd = VideoData(vf)

        for idx, result in enumerate(classify_video_shots(vd, args.topn)):
            frame = vd.keyframes[idx]
            print("[" + str(idx) + "] " + str(result) + " (" + frame.stem + ")")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
