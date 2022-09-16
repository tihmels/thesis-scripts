#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import skimage
import spacy
from PIL import Image
from alive_progress import alive_bar
from pytesseract import pytesseract, Output
from skimage.feature import match_template
from skimage.filters.edges import sobel
from skimage.filters.thresholding import try_all_threshold

from VideoData import get_date_time, VideoData, get_caption_file, get_shot_file
from utils.constants import TV_FILENAME_RE, TS_LOGO

TS_LOGO = np.array(Image.open(TS_LOGO).convert('L'))

spacy_de = spacy.load('de_core_news_sm')


def preprocess_caption_area(caption_area, is_nightly):
    if is_nightly:
        binary = caption_area > 205
        binary = skimage.morphology.binary_dilation(binary, footprint=skimage.morphology.diamond(1))
        return binary
    else:
        thresh = skimage.filters.thresholding.threshold_li(caption_area)
        binary = caption_area > thresh
        binary = skimage.morphology.binary_erosion(binary, footprint=skimage.morphology.diamond(1))
        return binary


def get_caption_area(frame, is_nightly, resize_factor=4):
    if is_nightly:
        return frame[-102 * resize_factor:-48 * resize_factor, 55 * resize_factor:]
    else:
        return frame[-48 * resize_factor:-18 * resize_factor, -425 * resize_factor:]


def is_nightly_version(vd: VideoData):
    ts_logo_area = (35, 225, 110, 250)

    center_frame_cropped = Image.open(vd.frames[int(vd.n_frames / 2)]).convert('L').crop(ts_logo_area)
    center_frame_cropped = np.array(center_frame_cropped)

    corr_coeff = match_template(center_frame_cropped, TS_LOGO)
    max_corr = np.max(corr_coeff)

    return max_corr > 0.9


def extract_caption_data(vd: VideoData, resize_factor=4):
    segments = vd.shots
    is_nightly = is_nightly_version(vd)

    for first_frame_idx, last_frame_idx, _ in segments:
        center_frame_path = vd.frames[int((first_frame_idx + last_frame_idx) / 2)]

        center_frame = Image.open(center_frame_path).convert('L')
        center_frame_resized = center_frame.resize(
            (center_frame.size[0] * resize_factor, center_frame.size[1] * resize_factor))
        center_frame_resized = np.array(center_frame_resized)

        caption_area = get_caption_area(center_frame_resized, is_nightly)
        caption_area = preprocess_caption_area(caption_area, is_nightly)

        custom_oem_psm_config = r'--psm 4 --oem 1'
        caption_data = pytesseract.image_to_data(caption_area, output_type=Output.DICT, lang='deu+eng',
                                                 config=custom_oem_psm_config)

        yield caption_data


def check_requirements(path: Path, skip_existing=True):
    assert path.parent.name == 'ts100'

    match = re.match(TV_FILENAME_RE, path.name)

    if match is None or not path.is_file():
        print(f'{path.name} does not exist or does not match TV-*.mp4 pattern.')
        return False

    shot_file = get_shot_file(path)

    if not shot_file.is_file():
        print(f'{path.name} has no detected shots.')
        return False

    if skip_existing and get_caption_file(path).exists():
        return False

    return True


if __name__ == "__main__":
    parser = ArgumentParser(description='Extracts banner captions from ts100 videos')
    parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+',
                        help='ts100 video files or directories to search for ts100 files')
    parser.add_argument('--overwrite', action='store_true', help='Re-extracts banner captions for all videos')
    args = parser.parse_args()

    video_files = []

    for file in args.files:
        if file.is_file() and check_requirements(file, not args.overwrite):
            video_files.append(file)
        elif file.is_dir():
            video_files.extend(
                [mp4_file for mp4_file in file.glob('*.mp4') if check_requirements(mp4_file, not args.overwrite)])

    assert len(video_files) > 0, 'No suitable files found'

    video_files.sort(key=get_date_time)

    print(f'Extracting banner captions from {len(video_files)} videos ... \n')

    for vf_idx, vf in enumerate(video_files):

        vd = VideoData(vf)

        with alive_bar(vd.n_shots, ctrl_c=False, title=f'[{vf_idx + 1}/{len(video_files)}] {vd}', length=20) as bar:

            captions = []

            for shot_idx, caption_data in enumerate(extract_caption_data(vd)):

                positive_confidence_indices = (np.array(caption_data['conf']) > 0.0).nonzero()

                if len(positive_confidence_indices[0]) < 1:
                    captions.append((shot_idx, "", -1))
                    bar()

                    continue

                strings = np.array(caption_data['text'])[positive_confidence_indices]
                confidences = np.array(caption_data['conf'])[positive_confidence_indices]
                blocks = np.array(caption_data['block_num'])[positive_confidence_indices]

                blocks_unique = np.unique(blocks, return_index=True, return_counts=True)
                blocks_unique = blocks_unique[1]

                s = [strings[blocks_unique[i]:blocks_unique[i + 1]] for i in range(0, len(blocks_unique) - 1)]
                s.append(strings[blocks_unique[-1]:])

                text = ' '.join(strings).strip()
                text = ' '.join(re.split("\s+", text, flags=re.UNICODE))

                mean_conf = np.mean(confidences) / 100

                doc = spacy_de(text)
                entities = doc.ents

                # if len(entities) == 1 and len(entities[0]) == len(doc):
                #     text = ''

                if mean_conf <= 0.8:
                    text = ''

                captions.append((shot_idx, text, mean_conf))

                bar()

            df = pd.DataFrame(data=captions, columns=['shot_idx', 'text', 'confidence'])
            df.to_csv(get_caption_file(vd), index=False)
