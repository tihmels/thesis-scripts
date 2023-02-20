import nltk
import numpy as np
import pandas as pd
import re
from PIL import Image
from functools import cached_property
from pathlib import Path
from skimage.feature import match_template
from typing import Union

from common.Schemas import SHOT_COLUMNS, BANNER_COLUMNS, STORY_COLUMNS, TRANSCRIPT_COLUMNS
from common.constants import TV_AUDIO_FILENAME_RE, STORY_AUDIO_FILENAME_RE, SHOT_AUDIO_FILENAME_RE, \
    STORY_TRANSCRIPT_FILENAME_RE, AUDIO_DIR, FRAME_DIR, KF_DIR, TRANSCRIPT_DIR, SM_DIR, TOPICS_FILENAME, \
    CAPTIONS_FILENAME, SHOT_CLASS_FILENAME, SHOT_FILENAME, TRANSCRIPT_FILENAME, STORY_FILENAME, TS_LOGO
from common.utils import frame_idx_to_time, time_to_datetime, Range, range_overlap

nltk.download('punkt')

from dataclasses import dataclass
from datetime import datetime


@dataclass
class BannerData:
    headline: str
    subline: str
    confidence: float

    @property
    def text(self):
        return ' '.join([self.headline, self.subline]) if self.subline else self.headline


@dataclass(frozen=True)
class TranscriptData:
    start: datetime.time
    end: datetime.time
    text: str
    color: str = None


def get_text(tds: [TranscriptData]):
    text = ' '.join([td.text.strip() for td in tds])
    return text


@dataclass
class ShotData:
    first_frame_idx: int
    last_frame_idx: int
    type: str = None

    @property
    def center_frame_idx(self):
        return int((self.first_frame_idx + self.last_frame_idx) / 2)

    @property
    def n_frames(self):
        return self.last_frame_idx - self.first_frame_idx + 1


@dataclass
class StoryData:
    ref_idx: int
    headline: str
    first_frame_idx: int
    last_frame_idx: int
    first_shot_idx: int
    last_shot_idx: int


class VAO:
    def __init__(self, path: Path):
        self.id: str = path.name.split('.')[0]
        self.path: Path = path
        self.date: datetime = get_date_time(path)
        self.dirs = self.Dirs(path)
        self.data = self.Data(path)

    @property
    def n_frames(self) -> int:
        return len(self.data.frames)

    @property
    def n_shots(self) -> int:
        return len(self.data.shots)

    @property
    def n_stories(self) -> int:
        return len(self.data.stories)

    @property
    def n_topics(self) -> int:
        return len(self.data.topics)

    @property
    def timecode(self) -> str:
        return self.id.split("-")[2]

    @property
    def date_str(self) -> str:
        return self.date.strftime("%Y%m%d")

    @property
    def is_summary(self) -> bool:
        return is_summary(self)

    @cached_property
    def is_nightly_version(self) -> bool:
        ts_logo_area = (35, 225, 110, 250)

        center_frame_cropped = Image.open(self.data.frames[int(self.n_frames / 2)]).convert('L').crop(ts_logo_area)
        center_frame_cropped = np.array(center_frame_cropped)

        ts_logo = Image.open(TS_LOGO).convert('L')

        corr_coeff = match_template(center_frame_cropped, np.array(ts_logo))
        max_corr = np.max(corr_coeff)

        return max_corr > 0.9

    @property
    def duration(self):
        return frame_idx_to_time(self.n_frames).replace(microsecond=0)

    def __str__(self):
        return str(self.path.relative_to(self.path.parent.parent)).split('.')[0]

    class Dirs:
        def __init__(self, path: Path):
            self.frame_dir: Path = get_frame_dir(path)
            self.keyframe_dir: Path = get_keyframe_dir(path)
            self.audio_dir: Path = get_audio_dir(path)
            self.transcripts_dir: Path = get_transcription_dir(path)
            self.sm_dir: Path = get_sm_dir(path)

    class Data:
        def __init__(self, path: Path):
            self._path: Path = path

        @cached_property
        def frames(self) -> [Path]:
            return sorted(get_frame_paths(self._path))

        @cached_property
        def keyframes(self) -> [Path]:
            return sorted(get_keyframe_paths(self._path))

        @cached_property
        def topics(self) -> [str]:
            return read_topics_from_file(get_topic_file(self._path))

        @cached_property
        def audio(self) -> Path:
            return get_main_audio_file(self._path)

        @cached_property
        def shots(self) -> [ShotData]:
            return read_shots_from_file(get_shot_file(self._path))

        @cached_property
        def banners(self) -> [BannerData]:
            return read_banner_captions_from_file(get_banner_caption_file(self._path))

        @cached_property
        def stories(self) -> [StoryData]:
            return read_stories_from_file(get_story_file(self._path))

        @cached_property
        def transcripts(self) -> [TranscriptData]:
            return read_transcript_from_file(get_main_transcript_file(self._path))

        def get_shot_transcripts(self, from_shot_idx, to_shot_idx=None) -> [TranscriptData]:
            to_shot_idx = to_shot_idx if to_shot_idx else from_shot_idx

            transcripts = []

            for shot in self.shots[from_shot_idx:to_shot_idx + 1]:
                from_time, to_time = frame_idx_to_time(shot.first_frame_idx), frame_idx_to_time(shot.last_frame_idx)
                from_time, to_time = time_to_datetime(from_time), time_to_datetime(to_time)

                shot_trans = [idx for idx, trans in enumerate(self.transcripts) if
                              range_overlap(Range(from_time, to_time),
                                            Range(time_to_datetime(trans.start), time_to_datetime(trans.end))) >=
                              0.6 * (time_to_datetime(trans.end) - time_to_datetime(trans.start)).total_seconds()]

                transcripts.extend(shot_trans)

            return list(dict.fromkeys(transcripts))

        def get_story_sentences(self, story_idx) -> [str]:
            story = self.stories[story_idx]

            idxs = self.get_shot_transcripts(story.first_shot_idx, story.last_shot_idx)

            text = ' '.join([self.transcripts[idx].text.strip() for idx in idxs]).strip()
            sentences = nltk.sent_tokenize(text, language='german')

            return sentences


VideoPathType = Union[VAO, Path]


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
        return 'ts100' in video.parts
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
    time_parser = lambda times: [datetime.strptime(time, '%H:%M:%S').time() for time in times]

    df = pd.read_csv(file,
                     usecols=lambda x: x in TRANSCRIPT_COLUMNS,
                     parse_dates=['start', 'end'],
                     date_parser=time_parser,
                     keep_default_na=False)

    return [TranscriptData(val[0], val[1], val[2], val[3] if 'color' in df else None) for val in df.values.tolist()]


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
    df = pd.read_csv(file, keep_default_na=False, usecols=STORY_COLUMNS)
    return [StoryData(val[0], val[1], val[2], val[3], val[5], val[6]) for val in df.values.tolist()]


def read_banner_captions_from_file(file: Path) -> [BannerData]:
    df = pd.read_csv(file, usecols=BANNER_COLUMNS, keep_default_na=False)
    return [BannerData(val[0], val[1], val[2]) for val in df.values.tolist()]


def read_shots_from_file(file: Path) -> [ShotData]:
    df = pd.read_csv(file, usecols=lambda x: x in SHOT_COLUMNS)
    return [ShotData(val[0], val[1], val[2] if 'type' in df else None) for val in df.values.tolist()]


def read_topics_from_file(file: Path) -> [str]:
    df = pd.read_csv(file, header=None, keep_default_na=False, decimal=',', usecols=[1])
    return [topic.strip() for sublist in df.values.tolist() for topic in sublist]
