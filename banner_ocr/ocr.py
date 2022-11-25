#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import skimage
from PIL import Image, ImageEnhance
from alive_progress import alive_bar
from pytesseract import pytesseract, Output
from skimage.feature import match_template
from skimage.filters.edges import sobel
from skimage.filters.thresholding import try_all_threshold

from common.VAO import get_date_time, VAO, get_banner_caption_file, get_shot_file, is_summary, \
    get_frame_dir, get_frame_paths
from common.constants import TV_FILENAME_RE, TS_LOGO

parser = ArgumentParser(description='Banner Caption Extraction')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+',
                    help="Tagesschau video file(s)")
parser.add_argument('--overwrite', action='store_false', dest='skip_existing',
                    help='Re-extracts banner captions for all videos')

TS_LOGO = np.array(Image.open(TS_LOGO).convert('L'))


def resize_frame(frame, resize_factor):
    height, width = frame.size
    return frame.resize((height * resize_factor, width * resize_factor))


def binarize_frame(frame, is_nightly):
    frame = np.array(frame)

    if is_nightly:
        binary = frame > 205
        # binary = skimage.morphology.binary_dilation(binary, footprint=skimage.morphology.diamond(1))
        return binary
    else:
        thresh = skimage.filters.thresholding.threshold_li(frame, initial_guess=150)
        # thresh = 165
        binary = frame > thresh
        # binary = skimage.morphology.binary_erosion(binary, footprint=skimage.morphology.diamond(2))
        return binary


def sharpen_frame(frame, factor):
    sharpness_enhancer = ImageEnhance.Sharpness(frame)
    return sharpness_enhancer.enhance(factor)


def crop_frame(frame, area):
    return frame.crop(area)


def is_nightly_version(vao: VAO):
    ts_logo_area = (35, 225, 110, 250)

    center_frame_cropped = Image.open(vao.data.frames[int(vao.n_frames / 2)]).convert('L').crop(ts_logo_area)
    center_frame_cropped = np.array(center_frame_cropped)

    corr_coeff = match_template(center_frame_cropped, TS_LOGO)
    max_corr = np.max(corr_coeff)

    return max_corr > 0.9


def extract_caption_data_from_frame(frame: Path, resize_factor, is_nightly, custom_oem_psm_config='--psm 4 --oem 1'):
    frame = Image.open(frame).convert('L')

    frame.save('/Users/tihmels/Desktop/out/1_original.jpg')

    width, height = frame.size
    area = (54, 168, width, 225) if is_nightly else (60, 224, width, 253)

    frame = crop_frame(frame, area)
    frame.save('/Users/tihmels/Desktop/out/2_area.jpg')

    frame = sharpen_frame(frame, 1.5)
    frame.save('/Users/tihmels/Desktop/out/3_sharpened.jpg')

    frame = resize_frame(frame, resize_factor)
    frame.save('/Users/tihmels/Desktop/out/4_resized.jpg')

    frame = binarize_frame(frame, is_nightly)
    Image.fromarray(frame).save('/Users/tihmels/Desktop/out/5_binarized.jpg')

    caption_data = pytesseract.image_to_data(frame, output_type=Output.DICT, lang='deu',
                                             config=custom_oem_psm_config)

    return caption_data


def extract_caption_data_from_shots(vao: VAO, resize_factor=3):
    shots = vao.data.shots

    is_nightly = is_nightly_version(vao)

    for shot in shots:
        center_frame = vao.data.frames[shot.center_frame_idx]

        image_data = extract_caption_data_from_frame(center_frame, resize_factor, is_nightly)

        yield image_data


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

        vao = VAO(vf)

        with alive_bar(vao.n_shots, ctrl_c=False,
                       title=f'[{vf_idx + 1}/{len(video_files)}] {vao} ' + ('☾' if is_nightly_version(vao) else '☼'),
                       length=20) as bar:

            captions = []

            for shot_idx, caption_data in enumerate(extract_caption_data_from_shots(vao)):

                positive_confidences = np.array(caption_data['conf']) > 0.0
                non_empty_texts = list(map(lambda idx: True if caption_data['text'][idx].strip() else False,
                                           range(0, len(positive_confidences))))

                positive_confidence_indices = (positive_confidences & non_empty_texts).nonzero()

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

                captions.append((headline, subline, int(np.rint(100 - (np.std(confidences))))))

                bar()

            df = pd.DataFrame(data=captions, columns=['headline', 'subline', 'confidence'])
            df.to_csv(get_banner_caption_file(vao), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
