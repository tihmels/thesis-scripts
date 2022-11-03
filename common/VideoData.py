import re
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd

from common.DataModel import CaptionData, TranscriptData, StoryData, ShotData, ShotClassificationData
from common.constants import TV_AUDIO_FILENAME_RE, STORY_AUDIO_FILENAME_RE, SHOT_AUDIO_FILENAME_RE, \
    STORY_TRANSCRIPT_FILENAME_RE, AUDIO_DIR, FRAME_DIR, KF_DIR, TRANSCRIPT_DIR, SM_DIR, TOPICS_FILENAME, \
    CAPTIONS_FILENAME, SHOT_CLASS_FILENAME, SHOT_FILENAME, TRANSCRIPT_FILENAME, STORY_FILENAME
from common.fs_utils import frame_idx_to_time, add_sec_to_time


class VideoData:
    def __init__(self, path: Path):
        self.id: str = path.stem
        self.path: Path = path
        self.date: datetime = get_date_time(path)
        self.frame_dir: Path = get_frame_dir(path)
        self.keyframe_dir: Path = get_keyframe_dir(path)
        self.audio_dir: Path = get_audio_dir(path)
        self.transcripts_dir: Path = get_transcription_dir(path)
        self.sm_dir: Path = get_sm_dir(path)
        self._topics: [str] = None
        self._frames: [Path] = None
        self._keyframes: [Path] = None
        self._classifications: [ShotClassificationData] = None
        self._shots: [ShotData] = None
        self._scenes: [StoryData] = None
        self._transcript: [TranscriptData] = None
        self._captions: [CaptionData] = None

    @property
    def topics(self) -> [str]:
        if self._topics is None:
            self._topics = read_topics_from_file(get_topic_file(self))
        return self._topics

    @property
    def audio(self) -> Path:
        return get_main_audio_file(self)

    @property
    def shots(self) -> [ShotData]:
        if self._shots is None:
            self._shots = read_shots_from_file(get_shot_file(self))
        return self._shots

    @property
    def captions(self) -> [CaptionData]:
        if self._captions is None:
            self._captions = read_banner_captions_from_file(get_banner_caption_file(self.path))
        return self._captions

    @property
    def stories(self) -> [StoryData]:
        if self._scenes is None:
            self._scenes = read_stories_from_file(get_story_file(self))
        return self._scenes

    @property
    def keyframes(self) -> [Path]:
        if self._keyframes is None:
            self._keyframes = sorted(get_keyframe_paths(self))
        return self._keyframes

    @property
    def frames(self) -> [Path]:
        if self._frames is None:
            self._frames = sorted(get_frame_paths(self))
        return self._frames

    @property
    def transcripts(self) -> [TranscriptData]:
        if self._transcript is None:
            self._transcript = read_transcript_from_file(get_main_transcript_file(self))
        return self._transcript

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def n_shots(self) -> int:
        return len(self.shots)

    @property
    def n_stories(self) -> int:
        return len(self.scenes)

    @property
    def timecode(self) -> str:
        return self.id.split("-")[2]

    @property
    def date_str(self) -> str:
        return self.date.strftime("%Y%m%d")

    @property
    def is_summary(self) -> bool:
        return is_summary(self)

    def get_shot_transcripts(self, from_idx, to_idx=None):
        from_shot, to_shot = self.shots[from_idx], self.shots[to_idx] if to_idx else self.shots[from_idx]
        from_time, to_time = frame_idx_to_time(from_shot.first_frame_idx), frame_idx_to_time(to_shot.last_frame_idx)

        from_time = from_time.replace(microsecond=0)
        to_time = add_sec_to_time(to_time, 1).replace(microsecond=0)

        return [trans for trans in self.transcript if from_time <= trans.start and trans.end <= to_time]

    def __str__(self):
        return str(self.path.relative_to(self.path.parent.parent)).split('.')[0]


VideoPathType = Union[VideoData, Path]


def get_date_time(video: VideoPathType):
    if isinstance(video, Path):
        date, time = video.name.split("-")[1:3]
        return datetime.strptime(date + time, "%Y%m%d%H%M")
    else:
        return get_date_time(video.path)


def get_data_dir(video: VideoPathType) -> Path:
    if isinstance(video, Path):
        return Path(video.parent, video.name.split(".")[0])
    else:
        return get_data_dir(video.path)


def get_audio_dir(video: VideoPathType) -> Path:
    if isinstance(video, Path):
        return Path(get_data_dir(video), AUDIO_DIR)
    else:
        return get_audio_dir(video.path)


def get_frame_dir(video: VideoPathType) -> Path:
    if isinstance(video, Path):
        return Path(get_data_dir(video), FRAME_DIR)
    else:
        return get_frame_dir(video.path)


def get_keyframe_dir(video: VideoPathType) -> Path:
    if isinstance(video, Path):
        return Path(get_data_dir(video), KF_DIR)
    else:
        return get_keyframe_dir(video.path)


def get_transcription_dir(video: VideoPathType) -> Path:
    if isinstance(video, Path):
        return Path(get_data_dir(video), TRANSCRIPT_DIR)
    else:
        return get_frame_dir(video.path)


def get_sm_dir(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), SM_DIR)
    else:
        return get_sm_dir(video.path)


def is_summary(video: VideoPathType):
    if isinstance(video, Path):
        return video.parent.name == 'ts100'
    else:
        return is_summary(video.path)


def get_topic_file(video: VideoPathType) -> Path:
    if isinstance(video, Path):
        return Path(get_data_dir(video), TOPICS_FILENAME)
    else:
        return get_topic_file(video.path)


def get_main_audio_file(video: VideoPathType) -> Path:
    if isinstance(video, Path):
        audio_dir = get_audio_dir(video)
        audio_files = [file for file in audio_dir.glob('*.wav') if re.match(TV_AUDIO_FILENAME_RE, file.name)]

        return audio_files[0] if audio_files else None
    else:
        return get_main_audio_file(video.path)


def get_shot_audio_files(video: VideoPathType) -> [Path]:
    if isinstance(video, Path):
        audio_dir = get_audio_dir(video)
        return sorted([audio for audio in audio_dir.glob('*.wav') if re.match(SHOT_AUDIO_FILENAME_RE, audio.name)])
    else:
        return get_shot_audio_files(video.path)


def get_story_audio_files(video: VideoPathType) -> [Path]:
    if isinstance(video, Path):
        audio_dir = get_audio_dir(video)
        return sorted([audio for audio in audio_dir.glob('*.wav') if re.match(STORY_AUDIO_FILENAME_RE, audio.name)])
    else:
        return get_story_audio_files(video.path)


def get_banner_caption_file(video: VideoPathType) -> Path:
    if isinstance(video, Path):
        return Path(get_data_dir(video), CAPTIONS_FILENAME)
    else:
        return get_banner_caption_file(video.path)


def get_shot_classification_file(video: VideoPathType) -> Path:
    if isinstance(video, Path):
        return Path(get_data_dir(video), SHOT_CLASS_FILENAME)
    else:
        return get_shot_classification_file(video.path)


def get_shot_file(video: VideoPathType) -> Path:
    if isinstance(video, Path):
        return Path(get_data_dir(video), SHOT_FILENAME)
    else:
        return get_shot_file(video.path)


def get_story_transcripts(video: VideoPathType) -> [Path]:
    if isinstance(video, Path):
        trans_dir = get_transcription_dir(video)
        return sorted([transcript for transcript in trans_dir.glob('*.txt') if
                       re.match(STORY_TRANSCRIPT_FILENAME_RE, transcript.name)])
    else:
        return get_story_audio_files(video.path)


def get_frame_paths(video: VideoPathType) -> [Path]:
    if isinstance(video, Path):
        frame_dir = get_frame_dir(video)
        return sorted(frame_dir.glob("frame_*.jpg"))
    else:
        return get_frame_paths(video.path)


def get_keyframe_paths(video: VideoPathType):
    if isinstance(video, Path):
        kf_dir = get_keyframe_dir(video)
        return sorted(kf_dir.glob("frame_*.jpg"))
    else:
        return get_keyframe_paths(video.path)


def get_main_transcript_file(video: VideoPathType):
    if isinstance(video, Path):
        return Path(get_data_dir(video), TRANSCRIPT_FILENAME)
    else:
        return get_main_transcript_file(video.path)


def read_transcript_from_file(file: Path) -> [TranscriptData]:
    columns = ['start', 'end', 'caption', 'color']

    time_parser = lambda times: [datetime.strptime(time, '%H:%M:%S').time() for time in times]

    df = pd.read_csv(file,
                     usecols=lambda x: x in columns,
                     parse_dates=['start', 'end'],
                     date_parser=time_parser,
                     keep_default_na=False)

    return list(
        map(lambda val: TranscriptData(val[0], val[1], val[2], val[3] if 'color' in df else None), df.values.tolist()))


def get_story_file(video: VideoPathType) -> Path:
    if isinstance(video, Path):
        return Path(get_data_dir(video), STORY_FILENAME)
    else:
        return get_story_file(video.path)


def get_xml_transcript_file(video: VideoPathType) -> Path:
    if isinstance(video, Path):
        date = video.name.split("-")[1]
        xml_files = [file for file in get_data_dir(video).iterdir() if
                     re.match(r'TV-' + date + r'-(\d{5}).xml', file.name)]
        return xml_files[0] if xml_files else None
    else:
        return get_xml_transcript_file(video.path)


def read_stories_from_file(file: Path) -> [StoryData]:
    df = pd.read_csv(file, usecols=['news_title',
                                    'first_frame_idx', 'last_frame_idx', 'n_frames',
                                    'first_shot_idx', 'last_shot_idx', 'n_shots',
                                    'from_ss', 'to_ss', 'total_ss'], keep_default_na=False)

    return list(map(lambda val: StoryData(val[0], val[1], val[2], val[4], val[5]), df.values.tolist()))


def read_banner_captions_from_file(file: Path) -> [CaptionData]:
    df = pd.read_csv(file, usecols=['headline', 'subline', 'confidence'], keep_default_na=False)
    return list(map(lambda val: CaptionData(val[0], val[1], val[2]), df.values.tolist()))


def read_shots_from_file(file: Path) -> [ShotData]:
    columns = ['first_frame_idx', 'last_frame_idx', 'type']
    df = pd.read_csv(file, usecols=lambda x: x in columns)
    return [ShotData(val[0], val[1], val[2] if 'type' in df else None) for val in df.values.tolist()]


def read_classifications_from_file(file: Path) -> [ShotClassificationData]:
    df = pd.read_csv(file, usecols=['class', 'prop'])
    return [ShotClassificationData(val[0], val[1]) for val in df.values.tolist()]


def read_topics_from_file(file: Path) -> [str]:
    df = pd.read_csv(file, skiprows=0, header=None, keep_default_na=False)
    return [topic.strip() for sublist in df.values.tolist() for topic in sublist]
