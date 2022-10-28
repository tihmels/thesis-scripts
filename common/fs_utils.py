import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from shutil import rmtree

from common.constants import SUMMARY_VIDEOS_PATH, TV_FILENAME_RE


def sec_to_time(seconds):
    return (datetime.min + timedelta(seconds=seconds)).time()


def add_sec_to_time(time, secs):
    fulldate = datetime(100, 1, 1, time.hour, time.minute, time.second)
    fulldate = fulldate + timedelta(seconds=secs)
    return fulldate.time()


def frame_idx_to_time(frame_idx):
    seconds = frame_idx / 25
    return sec_to_time(seconds)


def re_create_dir(path: Path):
    if path.is_dir():
        rmtree(path)

    path.mkdir(parents=True)


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
