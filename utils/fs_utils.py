import os
import re
from datetime import datetime
from pathlib import Path

from utils.constants import SUMMARY_VIDEOS_PATH, TV_FILENAME_RE


def read_segments_from_file(file):
    shots = []

    file = open(file, 'r')
    for line in file.readlines():
        first_index, last_index = [int(x.strip(' ')) for x in line.split(' ')]
        shots.append((first_index, last_index))

    return shots


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


def subdirs(root: str):
    sub_folders = [f.path for f in os.scandir(root) if f.is_dir()]
    for dir_name in list(sub_folders):
        sub_folders.extend(subdirs(dir_name))
    return [Path(f) for f in sub_folders] + [Path(root)]
