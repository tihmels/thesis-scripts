import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from shutil import rmtree

import cv2

from common.constants import SUMMARY_VIDEOS_PATH, TV_FILENAME_RE


def sec_to_frame_idx(second):
    return second * 25


def frame_idx_to_sec(idx):
    return idx / 25


def sec_to_time(seconds):
    return (datetime.min + timedelta(seconds=seconds)).time()


def read_images(img_paths: [Path], cvt_color=cv2.COLOR_BGR2RGB, resize=None):
    images = [cv2.imread(str(img)) for img in img_paths]
    images = [cv2.cvtColor(img, cvt_color) for img in images]

    if resize:
        images = [cv2.resize(img, resize) for img in images]

    return images


def add_sec_to_time(time, secs):
    fulldate = datetime(100, 1, 1, time.hour, time.minute, time.second)
    fulldate = fulldate + timedelta(seconds=secs)
    return fulldate.time()


def frame_idx_to_time(frame_idx):
    seconds = frame_idx_to_sec(frame_idx)
    return sec_to_time(seconds)


def create_dir(path: Path, rm_if_exist=False):
    if path.is_dir() and rm_if_exist:
        rmtree(path)

    path.mkdir(parents=True, exist_ok=False if rm_if_exist else True)


def get_summary_videos():
    return [file for file in Path(SUMMARY_VIDEOS_PATH).glob("*.mp4") if re.match(TV_FILENAME_RE, file.name)]


def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)
