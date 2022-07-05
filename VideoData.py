import re
from datetime import datetime
from pathlib import Path
from typing import Union

from utils.constants import AUDIO_FILENAME_RE, SHOT_FILENAME_RE


class VideoData:
    def __init__(self, path: Path):
        self.id: str = path.stem
        self.path: Path = path
        self.date: datetime = get_date_time(path)
        self.frame_dir: Path = get_frame_dir(path)
        self.keyframe_dir: Path = get_kf_dir(path)
        self.audio_dir: Path = get_audio_dir(path)
        self.sm_dir: Path = get_sm_dir(path)
        self._frames: [Path] = None
        self._kfs: [Path] = None
        self._shots: [(int, int)] = None
        self._audio_file = None
        self._audio_shots: [Path] = None

    @property
    def shots(self):
        if self._shots is None:
            self._shots = read_shots_from_file(get_shot_file(self.path))
        return self._shots

    @property
    def kfs(self):
        if self._kfs is None:
            self._kfs = sorted(get_keyframe_paths(self.path))
        return self._kfs

    @property
    def frames(self):
        if self._frames is None:
            self._frames = sorted(get_frame_paths(self.path))
        return self._frames

    @property
    def audio_file(self):
        if self._audio_file is None:
            self._audio_file = get_audio_file(self.path)
        return self._audio_file

    @property
    def audio_shots(self):
        if self._audio_shots is None:
            self._audio_shots = sorted(get_audio_shots(self.path))
        return self._audio_file

    @property
    def n_frames(self):
        return len(self.frames)

    @property
    def n_shots(self):
        return len(self.shots)

    @property
    def timecode(self):
        return self.id.split("-")[2]

    @property
    def date_str(self):
        return self.date.strftime("%Y%m%d")

    def __str__(self):
        return str(self.path.relative_to(self.path.parent.parent)).split('.')[0]


VideoPathType = Union[VideoData, Path]


def get_data_dir(video: VideoPathType):
    if isinstance(video, Path):
        return Path(video.parent, video.name.split(".")[0])
    else:
        return get_data_dir(video.path)


def get_audio_dir(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), "audio")
    else:
        return get_audio_dir(video.path)


def get_audio_file(video: VideoPathType):
    if isinstance(video, Path):
        files = [f for f in get_audio_dir(video).glob('*.wav') if re.match(AUDIO_FILENAME_RE, f.name)]
        if len(files) == 1:
            return files[0]
        else:
            return None
    else:
        return get_audio_file(video.path)


def get_audio_shots(video: VideoPathType):
    if isinstance(video, Path):
        return [audio for audio in get_audio_dir(video).glob('*.wav') if re.match(SHOT_FILENAME_RE, audio.name)]
    else:
        return get_audio_shots(video.path)


def get_sm_dir(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), "sm")
    else:
        return get_sm_dir(video.path)


def get_topic_file(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), "topics.csv")
    else:
        return get_topic_file(video.path)


def get_kf_dir(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), "kfs")
    else:
        return get_kf_dir(video.path)


def get_date_time(video: VideoPathType):
    if isinstance(video, Path):
        date, time = video.name.split("-")[1:3]
        return datetime.strptime(date + time, "%Y%m%d%H%M")
    else:
        return get_date_time(video.path)


def get_shot_file(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), "shots.txt")
    else:
        return get_shot_file(video.path)


def get_frame_dir(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), "frames")
    else:
        return get_frame_dir(video.path)


def get_frame_paths(video: VideoPathType):
    if isinstance(video, Path):
        frame_dir = get_frame_dir(video)
        return list(frame_dir.glob("frame_*.jpg"))
    else:
        return get_frame_paths(video.path)


def get_keyframe_paths(video: VideoPathType):
    if isinstance(video, Path):
        frame_dir = get_kf_dir(video)
        return list(frame_dir.glob("frame_*.jpg"))
    else:
        return get_keyframe_paths(video.path)


def get_feature_file(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), "features.h5")
    else:
        return get_feature_file(video.path)


def read_shots_from_file(file: Path):
    shots = []

    if file.is_file():
        file = open(file, 'r')
        for line in file.readlines():
            first_index, last_index = [int(x.strip(' ')) for x in line.split(' ')]
            shots.append((first_index, last_index))
        return shots
    else:
        return None
