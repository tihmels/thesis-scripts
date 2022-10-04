import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorflow as tf
from alive_progress import alive_bar

from common.VideoData import get_keyframe_dir, get_date_time, VideoData
from common.constants import TV_FILENAME_RE


def detect_anchorshots(vd: VideoData):
    model = tf.keras.models.load_model(Path(Path(__file__).parent.resolve(), 'model', 'ts_anchorshot_model'))
    model.load_weights('tl_model_v1.weights.best.hdf5')

    classes = ['anchor', 'non-anchor']

    keyframes = [tf.keras.preprocessing.image.load_img(img, target_size=(224, 224)) for img in vd.keyframes]
    keyframes = [tf.keras.preprocessing.image.img_to_array(img) for img in keyframes]
    keyframes = [np.expand_dims(img, axis=0) for img in keyframes]

    for kf in keyframes:
        frame = tf.keras.applications.vgg19.preprocess_input(kf)
        prediction = model.predict(frame)[0]

        result = [(classes[i], float(prediction[i]) * 100.0) for i in range(len(prediction))]
        result.sort(reverse=True, key=lambda x: x[1])

        yield result[0]


def check_requirements(path: Path, skip_existing=False):
    match = re.match(TV_FILENAME_RE, path.name)

    if match is None or not path.is_file():
        print(f'{path.name} does not exist or does not match TV-*.mp4 pattern.')
        return False

    kf_dir = get_keyframe_dir(path)

    if not kf_dir.is_dir() or len(list(kf_dir.glob("*.jpg"))) < 1:
        return False

    return True


if __name__ == "__main__":
    parser = ArgumentParser('Anchorshot Detection using pretrained Keras Model')
    parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    video_files = []

    for file in args.files:
        if file.is_file() and check_requirements(file, not args.overwrite):
            video_files.append(file)
        elif file.is_dir():
            video_files.extend([video for video in file.glob('*.mp4') if check_requirements(video, not args.overwrite)])

    assert len(video_files) > 0

    video_files.sort(key=get_date_time)

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        with alive_bar(vd.n_shots, ctrl_c=False, title=f'[{idx + 1}/{vd.n_shots}] {vd}', length=20) as bar:
            for result in detect_anchorshots(vd):
                print(result)
                bar()
