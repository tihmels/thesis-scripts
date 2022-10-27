from dataclasses import dataclass, field
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
