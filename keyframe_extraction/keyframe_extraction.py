#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

from VideoData import VideoData
from utils.constants import TV_FILENAME_RE
from utils.fs_utils import get_frame_dir, get_shot_file, get_date_time, get_kf_dir, read_segments_from_file


def convolution(image, kernel, padding=0, strides=1):
    kernel = np.flipud(np.fliplr(kernel))

    kernel_shape_x, kernel_shape_y = kernel.shape[0], kernel.shape[1]
    img_shape_x, img_shape_y = image.shape[0], image.shape[1]

    output_x = int(((img_shape_x - kernel_shape_x + 2 * padding) / strides) + 1)
    output_y = int(((img_shape_y - kernel_shape_y + 2 * padding) / strides) + 1)

    output = np.zeros((output_x, output_y))

    if padding != 0:
        image_padded = np.zeros((img_shape_x + padding * 2, img_shape_y + padding * 2))
        image_padded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        image_padded = image

    for y in range(img_shape_y):

        if y > img_shape_y - kernel_shape_y:
            break

        if y % strides == 0:
            for x in range(img_shape_x):
                if x > img_shape_x - kernel_shape_x:
                    break
                try:
                    if x % strides == 0:
                        output[x, y] = (kernel * image_padded[x: x + kernel_shape_x, y: y + kernel_shape_y]).sum()
                except:
                    break

    return output


def detect_keyframes(vd: VideoData, skip_existing: False):
    segments = vd.segments

    keyframe_indices = []

    for idx, (seg_start, seg_end) in enumerate(segments):

        print(f'[{idx}/{len(segments)}|{seg_start}-{seg_end}]:', end=' ')

        if skip_existing and Path(vd.kf_dir, 'shot_' + str(idx) + '.jpg').is_file():
            print(f'already extracted ...')
            continue

        frames = [Image.open(frame) for frame in vd.frames[seg_start:seg_end]]
        np_frames = [np.asarray(frame.convert('L')) for frame in frames]

        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

        gradient_magnitudes = []

        for frame in np_frames:
            grad_x = convolution(frame, sobel_x, padding=2)
            grad_y = convolution(frame, sobel_y, padding=2)

            grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
            gradient_magnitudes.append(grad_mag)

        def mean(gradient):
            numerator = np.sum(gradient)
            denominator = gradient.shape[0]
            return numerator / denominator

        def std(gradient, mean):
            z_grad = np.square(gradient - mean)
            numerator = np.sum(z_grad)
            denominator = gradient.shape[0]
            return np.sqrt(numerator / denominator)

        means = [mean(grad_mag) for grad_mag in gradient_magnitudes]
        stds = [std(grad_mag, mean) for grad_mag, mean in
                zip(gradient_magnitudes, means)]

        zgms = [((grad_mag - mean) / std) for grad_mag, mean, std in zip(gradient_magnitudes, means, stds)]

        zgm_means = [mean(zgm) for zgm in zgms]
        zgm_stds = [std(zgm, mean) for zgm, mean in zip(zgms, zgm_means)]

        cvs = [(zgm_std / zgm_mean) for zgm_mean, zgm_std in zip(zgm_means, zgm_stds)]

        keyframe_idx = np.argmax(cvs)
        keyframe_indices.append(keyframe_idx)

        print(keyframe_idx + seg_start)

        yield frames[keyframe_idx]


def check_requirements(path: Path, skip_existing: False):
    match = re.match(TV_FILENAME_RE, path.name)

    if match is None or not path.is_file():
        print(f'{path.name} does not exist or does not match TV-*.mp4 pattern.')
        return False

    frame_dir = get_frame_dir(path)

    if not frame_dir.is_dir() or len(list(frame_dir.glob('frame_*.jpg'))) == 0:
        print(f'{file.name} no frames have been extracted.')
        return False

    shot_file = get_shot_file(path)

    if not shot_file.exists():
        print(f'{path.name} has no detected shots.')
        return False

    kf_dir = get_kf_dir(path)

    if skip_existing and kf_dir.exists() and len(list(kf_dir.glob('shot_*.jpg'))) == len(
            read_segments_from_file(shot_file)):
        print(f'{path.name} has already keyframes extracted. Skipping ...')
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
    parser.add_argument('-s', '--skip', action='store_true', help="skip keyframe extraction if already exist")
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    video_files = []

    for file in args.files:
        if file.is_file() and check_requirements(file, args.skip):
            video_files.append(file)
        elif file.is_dir():
            video_files.extend([video for video in file.glob('*.mp4') if check_requirements(video, args.skip)])

    assert len(video_files) > 0

    video_files.sort(key=get_date_time)

    print(f'Keyframe extraction for ({len(video_files)} videos)\n')

    for idx, video in enumerate(video_files):
        vd = VideoData(video)

        vd.kf_dir.mkdir(exist_ok=True)

        print(f'({idx}/{len(video_files)}) {vd.id}')

        for idx, kf in enumerate(detect_keyframes(vd, args.skip)):
            kf.save(Path(vd.kf_dir, "shot_" + str(idx) + ".jpg"))
