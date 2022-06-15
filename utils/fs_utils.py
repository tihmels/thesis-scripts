import os
from pathlib import Path


def subdirs(root: str):
    sub_folders = [f.path for f in os.scandir(root) if f.is_dir()]
    for dir_name in list(sub_folders):
        sub_folders.extend(subdirs(dir_name))
    return [Path(f) for f in sub_folders] + [Path(root)]
