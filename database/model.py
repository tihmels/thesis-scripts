import datetime
from abc import ABC
from typing import List, Optional

from redis_om import Field, JsonModel

from database import red


class BaseModel(JsonModel, ABC):
    class Meta:
        database = red
        orm_mode = True
        arbitrary_types_allowed = True
        extra = "allow"
        global_key_prefix = 'tsv'


class EmbeddedBaseModel(BaseModel, ABC):
    class Meta:
        embedded = True


class Topic(EmbeddedBaseModel):
    title: str


class Banner(EmbeddedBaseModel):
    text: str
    confidence: int


class Transcript(EmbeddedBaseModel):
    from_time: datetime.time
    to_time: datetime.time
    text: str


class Sentence(EmbeddedBaseModel):
    text: str


class Shot(EmbeddedBaseModel):
    first_frame_idx: int
    last_frame_idx: int
    duration: datetime.time
    text: str
    keyframe: str
    type: Optional[str]


class ShortShot(Shot):
    banner: Banner


class Story(EmbeddedBaseModel):
    headline: str
    start: datetime.time
    end: datetime.time
    duration: datetime.time
    shots: List[Shot]
    sentences: List[Sentence]


class MainStory(Story):
    topic: Topic


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
    topics: List[Topic]

    class Meta:
        model_key_prefix = 'ts15'


class VideoRef(EmbeddedBaseModel):
    ref_pk: str = Field(index=True)
    temp_dist: int = Field(index=True, sortable=True)
    similarity: int = Field(index=True, sortable=True, default=-1)


class ShortVideo(VideoBaseModel):
    is_nightly: int = Field(index=True)

    pre_main: Optional[VideoRef]
    suc_main: Optional[VideoRef]

    class Meta:
        model_key_prefix = 'ts100'
