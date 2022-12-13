import datetime
from abc import ABC
from redis_om import Field, JsonModel
from typing import List, Optional

from database import db


class BaseModel(JsonModel, ABC):
    class Meta:
        database = db
        orm_mode = True
        arbitrary_types_allowed = True
        extra = "allow"
        global_key_prefix = 'tsv'


class EmbeddedBaseModel(BaseModel, ABC):
    class Meta:
        embedded = True


class Headline(EmbeddedBaseModel):
    text: str


class Banner(EmbeddedBaseModel):
    text: str
    confidence: int


class Transcript(EmbeddedBaseModel):
    from_time: datetime.time
    to_time: datetime.time
    text: str


class Shot(EmbeddedBaseModel, ABC):
    first_frame_idx: int
    last_frame_idx: int
    duration: datetime.time
    transcript: str
    keyframe: str


class MainShot(Shot):
    type: str

    class Meta:
        model_key_prefix = 'shot'


class ShortShot(Shot):
    banner: Banner

    class Meta:
        model_key_prefix = 'shot'


class Story(EmbeddedBaseModel):
    headline: str
    first_frame_idx: int
    last_frame_idx: int
    start: datetime.time
    end: datetime.time
    duration: datetime.time
    frames: List[str]
    shots: List[Shot]
    sentences: List[str]

    class Meta:
        model_key_prefix = 'story'


class TopicCluster(BaseModel):
    index: int = Field(index=True, sortable=True)
    features: int = Field(index=True, default=0)
    keywords: List[str]

    ts15s: List[Story]
    ts100s: List[Story]

    class Meta:
        model_key_prefix = 'cluster'


class NodeBaseModel(BaseModel, ABC):
    pre_pk: str = Field(index=True, default="")
    suc_pk: str = Field(index=True, default="")


class VideoBaseModel(NodeBaseModel, ABC):
    path: str
    date: datetime.date
    time: datetime.time
    duration: datetime.time
    timestamp: int = Field(index=True, sortable=True)
    shots: List[Shot]
    stories: List[Story]
    transcripts: List[Transcript]


class MainVideo(VideoBaseModel):
    topics: List[str]

    class Meta:
        model_key_prefix = 'ts15'


class VideoRef(EmbeddedBaseModel):
    ref_pk: str = Field(index=True)
    temp_dist: int = Field(index=True, sortable=True)
    similarity: int = Field(index=True, sortable=True, default=-1)


class ShortVideo(VideoBaseModel):
    pre_main: Optional[VideoRef]
    suc_main: Optional[VideoRef]

    class Meta:
        model_key_prefix = 'ts100'
