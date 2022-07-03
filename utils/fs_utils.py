import logging
import os
import re
from datetime import datetime
from pathlib import Path

from utils.constants import SUMMARY_VIDEOS_PATH, TV_FILENAME_RE


def get_audio_dir(video: Path):
    return Path(get_data_dir(video), "audio")


def get_sm_dir(video: Path):
    return Path(get_data_dir(video), "sm")


def get_kf_dir(video: Path):
    return Path(get_data_dir(video), "kfs")


def get_date_time(video: Path):
    date, time = video.name.split("-")[1:3]
    return datetime.strptime(date + time, "%Y%m%d%H%M")


def get_data_dir(video: Path):
    return Path(video.parent, video.name.split(".")[0])


def get_shot_file(video: Path):
    return Path(get_data_dir(video), "shots.txt")


def get_frame_dir(video: Path):
    return Path(get_data_dir(video), "frames")


def get_summary_videos():
    return [file for file in Path(SUMMARY_VIDEOS_PATH).glob("*.mp4") if re.match(TV_FILENAME_RE, file.name)]


def read_segments_from_file(file):
    shots = []

    file = open(file, 'r')
    for line in file.readlines():
        first_index, last_index = [int(x.strip(' ')) for x in line.split(' ')]
        shots.append((first_index, last_index))

    return shots


def subdirs(root: str):
    sub_folders = [f.path for f in os.scandir(root) if f.is_dir()]
    for dir_name in list(sub_folders):
        sub_folders.extend(subdirs(dir_name))
    return [Path(f) for f in sub_folders] + [Path(root)]


# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


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
