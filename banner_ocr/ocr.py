#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import skimage
import spacy
from PIL import Image, ImageEnhance
from alive_progress import alive_bar
from pytesseract import pytesseract, Output
from skimage.feature import match_template
from skimage.filters.edges import sobel
from skimage.filters.thresholding import try_all_threshold

from common.VideoData import get_date_time, VideoData, get_banner_caption_file, get_shot_file, is_summary, \
    get_frame_dir, get_frame_paths
from common.constants import TV_FILENAME_RE, TS_LOGO

parser = ArgumentParser(description='Banner Caption Extraction')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+',
                    help="Tagesschau video file(s)")
parser.add_argument('--overwrite', action='store_false', dest='skip_existing',
                    help='Re-extracts banner captions for all videos')

TS_LOGO = np.array(Image.open(TS_LOGO).convert('L'))

spacy_de = spacy.load('de_core_news_sm')


def preprocess_caption_area(caption_area, is_nightly):
    if is_nightly:
        binary = caption_area > 205
        binary = skimage.morphology.binary_dilation(binary, footprint=skimage.morphology.diamond(1))
        return binary
    else:
        Image.fromarray(caption_area).save("/Users/tihmels/Desktop/before_preprocess.jpg")
        thresh = skimage.filters.thresholding.threshold_li(caption_area)
        binary = caption_area > thresh
        # binary = skimage.morphology.binary_erosion(binary, footprint=skimage.morphology.diamond(1))
        Image.fromarray(binary).save("/Users/tihmels/Desktop/after_preprocess.jpg")
        return binary


def get_caption_area(frame, is_nightly, resize_factor=4):
    if is_nightly:
        return frame[-102 * resize_factor:-48 * resize_factor, 55 * resize_factor:]
    else:
        return frame[-46 * resize_factor:-17 * resize_factor, -420 * resize_factor:]


def is_nightly_version(vd: VideoData):
    ts_logo_area = (35, 225, 110, 250)

    center_frame_cropped = Image.open(vd.frames[int(vd.n_frames / 2)]).convert('L').crop(ts_logo_area)
    center_frame_cropped = np.array(center_frame_cropped)

    corr_coeff = match_template(center_frame_cropped, TS_LOGO)
    max_corr = np.max(corr_coeff)

    return max_corr > 0.9


def extract_caption_data_per_shot(vd: VideoData, resize_factor=4):
    segments = vd.shots

    is_nightly = is_nightly_version(vd)

    for first_frame_idx, last_frame_idx in segments:
        center_frame_index = int(((first_frame_idx + last_frame_idx) / 2))
        center_frame_path = vd.frames[center_frame_index]

        center_frame = Image.open(center_frame_path).convert('L')

        center_frame.save("/Users/tihmels/Desktop/before.jpg")

        sharpness_enhancer = ImageEnhance.Sharpness(center_frame)
        center_frame = sharpness_enhancer.enhance(2.0)

        center_frame.save("/Users/tihmels/Desktop/after.jpg")

        center_frame_resized = center_frame.resize(
            (center_frame.size[0] * resize_factor, center_frame.size[1] * resize_factor))
        center_frame_resized = np.array(center_frame_resized)

        caption_area = get_caption_area(center_frame_resized, is_nightly)
        caption_area = preprocess_caption_area(caption_area, is_nightly)

        custom_oem_psm_config = r'--psm 4 --oem 1'
        caption_data = pytesseract.image_to_data(caption_area, output_type=Output.DICT, lang='deu',
                                                 config=custom_oem_psm_config)

        yield caption_data


def check_requirements(video: Path):
    assert is_summary(video)

    if not re.match(TV_FILENAME_RE, video.name):
        print(f'{video.name} does not match TV-*.mp4 pattern.')
        return False

    frame_dir = get_frame_dir(video)

    if not frame_dir.is_dir() or not len(get_frame_paths(video)) > 0:
        print(f'{video.name} has no extracted frames.')
        return False

    shot_file = get_shot_file(video)

    if not shot_file.is_file():
        print(f'{video.name} has no detected shots.')
        return False

    return True


def was_processed(video: Path):
    return get_banner_caption_file(video).is_file()


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0

    video_files = sorted(video_files, key=get_date_time)

    print(f'Extracting banner captions from {len(video_files)} videos ... \n')

    for vf_idx, vf in enumerate(video_files):

        vd = VideoData(vf)

        with alive_bar(vd.n_shots, ctrl_c=False,
                       title=f'[{vf_idx + 1}/{len(video_files)}] {vd} ' + ('☾' if is_nightly_version(vd) else '☼'),
                       length=20) as bar:

            captions = []

            for shot_idx, caption_data in enumerate(extract_caption_data_per_shot(vd)):

                positive_confidence_indices = (np.array(caption_data['conf']) > 0.0).nonzero()

                if len(positive_confidence_indices[0]) == 0:
                    captions.append(("", "", -1))
                    bar()

                    continue

                strings = np.array(caption_data['text'])[positive_confidence_indices]
                confidences = np.array(caption_data['conf'])[positive_confidence_indices]

                blocks = np.array(caption_data['block_num'])[positive_confidence_indices]
                lines = np.array(caption_data['line_num'])[positive_confidence_indices]

                blocks_vs_lines = [max(block, line) for block, line in zip(blocks, lines)]

                unique_blocks_indices = np.unique(blocks_vs_lines, return_index=True)[1]

                rows = [strings[unique_blocks_indices[i]:unique_blocks_indices[i + 1]] for i in
                        range(0, len(unique_blocks_indices) - 1)]
                rows.append(strings[unique_blocks_indices[-1]:])

                rows = [row.tolist() for row in rows]

                headline, *sublines = rows

                headline = ' '.join(headline).strip()
                subline = ' '.join([sub for subline in sublines for sub in subline]).strip()

                mean_conf = np.mean(confidences) / 100

                captions.append((headline, subline, np.around(mean_conf, 2)))

                bar()

            df = pd.DataFrame(data=captions, columns=['headline', 'subline', 'confidence'])
            df.to_csv(get_banner_caption_file(vd), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
