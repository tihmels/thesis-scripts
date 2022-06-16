import os
from pathlib import Path


def get_data_dir(video: Path):
    return Path(video.parent, video.name.split(".")[0])


def get_shot_file(video: Path):
    return Path(get_data_dir(video), "shots.txt")


def get_frame_dir(video: Path):
    return Path(get_data_dir(video), "frames")


def subdirs(root: str):
    sub_folders = [f.path for f in os.scandir(root) if f.is_dir()]
    for dir_name in list(sub_folders):
        sub_folders.extend(subdirs(dir_name))
    return [Path(f) for f in sub_folders] + [Path(root)]
