from dataclasses import dataclass
from datetime import datetime
from enum import Enum


@dataclass
class CaptionData:
    headline: str
    subline: str
    confidence: float

    @property
    def text(self):
        return ' '.join([self.headline, self.subline]) if self.subline else self.headline


@dataclass
class TranscriptData:
    start: datetime.time
    end: datetime.time
    text: str
    color: str = None


class ShotType(Enum):
    ANCHOR = 1
    NEWS = 2


@dataclass
class ShotData:
    first_frame_idx: int
    last_frame_idx: int
    type: str

    def center_frame_idx(self):
        return int((self.first_frame_idx + self.last_frame_idx) / 2)

    def n_frames(self):
        return self.last_frame_idx - self.first_frame_idx + 1


@dataclass
class StoryData:
    title: str
    first_frame_idx: int
    last_frame_idx: int
    first_shot_idx: int
    last_shot_idx: int


@dataclass
class ShotClassificationData:
    clazz: str
    prop: float
