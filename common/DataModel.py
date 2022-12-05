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


@dataclass
class TranscriptData:
    start: datetime.time
    end: datetime.time
    text: str
    color: str = None


def get_text(tds: [TranscriptData]):
    return ' '.join([td.text for td in tds])


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
