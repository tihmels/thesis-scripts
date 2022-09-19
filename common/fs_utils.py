import logging
import os
import re
from pathlib import Path

from common.constants import SUMMARY_VIDEOS_PATH, TV_FILENAME_RE


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
