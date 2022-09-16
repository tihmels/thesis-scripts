import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd


class VideoData:
    def __init__(self, path: Path):
        self.id: str = path.stem
        self.path: Path = path
        self.date: datetime = get_date_time(path)
        self.frame_dir: Path = get_frame_dir(path)
        self.keyframe_dir: Path = get_keyframe_dir(path)
        self.audio_dir: Path = get_audio_dir(path)
        self.sm_dir: Path = get_sm_dir(path)
        self._shots: [(int, int)] = None
        self._scenes = None
        self._frames: [Path] = None
        self._keyframes: [Path] = None
        self._captions: [Path] = None
        self._audio: Path = None
        self._audio_shots: [Path] = None

    @property
    def shots(self):
        if self._shots is None:
            self._shots = read_shots_from_file(get_shot_file(self.path))
        return self._shots

    @property
    def captions(self):
        if self._captions is None:
            self._captions = read_captions_from_file(get_caption_file(self.path), is_summary(self))
        return self._captions

    @property
    def scenes(self):
        if self._scenes is None:
            self._scenes = read_scenes_from_file(get_scene_file(self.path))
        return self._scenes

    @property
    def keyframes(self):
        if self._keyframes is None:
            self._keyframes = sorted(get_keyframe_paths(self.path))
        return self._keyframes

    @property
    def frames(self):
        if self._frames is None:
            self._frames = sorted(get_frame_paths(self.path))
        return self._frames

    @property
    def audio(self):
        if self._audio is None:
            self._audio = get_audio_file(self.path)
        return self._audio

    @property
    def audio_shots(self):
        if self._audio_shots is None:
            self._audio_shots = [path for path in sorted(get_audio_shot_paths(self.path))]
        return self._audio

    @property
    def n_frames(self):
        return len(self.frames)

    @property
    def n_shots(self):
        return len(self.shots)

    @property
    def n_stories(self):
        return len(self.scenes)

    @property
    def timecode(self):
        return self.id.split("-")[2]

    @property
    def date_str(self):
        return self.date.strftime("%Y%m%d")

    @property
    def is_summary(self):
        return is_summary(self)

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
        files = [file for file in get_audio_dir(video).glob('*.wav') if
                 re.match(r'TV-(\d{8})-(\d{4})-(\d{4}).webs.h264.wav', file.name)]
        if len(files) == 1:
            return files[0]
        else:
            return None
    else:
        return get_audio_file(video.path)


def get_audio_shot_paths(video: VideoPathType):
    if isinstance(video, Path):
        return [audio for audio in get_audio_dir(video).glob('*.wav') if re.match(r'shot_\d*.wav', audio.name)]
    else:
        return get_audio_shot_paths(video.path)


def get_sm_dir(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), "sm")
    else:
        return get_sm_dir(video.path)


def is_summary(video: VideoPathType):
    if isinstance(video, Path):
        return video.parent.name == 'ts100'
    else:
        return is_summary(video.path)


def get_caption_file(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), "captions.csv")
    else:
        return get_caption_file(video.path)


def get_keyframe_dir(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), "kfs")
    else:
        return get_keyframe_dir(video.path)


def get_date_time(video: VideoPathType):
    if isinstance(video, Path):
        date, time = video.name.split("-")[1:3]
        return datetime.strptime(date + time, "%Y%m%d%H%M")
    else:
        return get_date_time(video.path)


def get_shot_file(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), "shots.csv")
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
        frame_dir = get_keyframe_dir(video)
        return list(frame_dir.glob("frame_*.jpg"))
    else:
        return get_keyframe_paths(video.path)


def get_feature_file(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), "features.h5")
    else:
        return get_feature_file(video.path)


def get_transcript_file(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), "transcript.json")
    else:
        return get_transcript_file(video.path)


def get_scene_file(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), "stories.csv")
    else:
        return get_scene_file(video.path)


def read_scenes_from_file(file: Path):
    df = pd.read_csv(file, usecols=['news_title',
                                    'first_frame_idx', 'last_frame_idx', 'n_frames',
                                    'first_shot_idx', 'last_shot_idx', 'n_shots',
                                    'from_ss', 'to_ss', 'total_ss'], keep_default_na=False)
    return df


def read_captions_from_file(file: Path, is_summary: bool):
    if is_summary:
        df = pd.read_csv(file, usecols=['shot_idx', 'text', 'confidence'], keep_default_na=False)
        return list(df.to_records(index=False))
    else:
        with open(file, 'r') as file:
            reader = csv.reader(file)
            return [row for row in reader]


def read_shots_from_file(file: Path):
    df = pd.read_csv(file, usecols=['first_frame_idx', 'last_frame_idx', 'n_frames'])
    return list(df.to_records(index=False))
