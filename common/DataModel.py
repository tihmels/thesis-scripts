from dataclasses import dataclass
from datetime import datetime


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


@dataclass
class ShotData:
    first_frame_idx: int
    last_frame_idx: int


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
