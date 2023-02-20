import datetime
from abc import ABC
from typing import List, Optional

from redis_om import Field, JsonModel

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


class Banner(EmbeddedBaseModel):
    headline: str
    subheadline: str
    confidence: int


class Transcript(EmbeddedBaseModel):
    from_time: datetime.time
    to_time: datetime.time
    text: str


class Shot(EmbeddedBaseModel, ABC):
    first_frame_idx: int
    last_frame_idx: int
    duration: datetime.time
    transcripts: List[Transcript]
    keyframe: str


class BannerShot(Shot):
    banner: Banner


class Story(EmbeddedBaseModel):
    headline: str = Field(index=True)
    video: str = Field(index=True)
    type: str = Field(index=True)
    first_frame_idx: int
    last_frame_idx: int
    timestamp: int
    start_time: datetime.time
    end_time: datetime.time
    duration: datetime.time
    frames: List[str]
    shots: List[Shot]
    sentences: List[str]

    class Meta:
        model_key_prefix = 'story'


class TopicCluster(BaseModel):
    index: int = Field(index=True, sortable=True)
    keywords: List[str]

    n_ts15: int
    n_ts100: int

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
    frames: List[str]
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


class ShortVideo(VideoBaseModel):
    pre_main: Optional[VideoRef]
    suc_main: Optional[VideoRef]

    class Meta:
        model_key_prefix = 'ts100'


RAI_HEADLINE_PREFIX = 'tensor:headline:'
RAI_TOPIC_PREFIX = 'tensor:topic:'
RAI_TEXT_PREFIX = 'tensor:mil-nce:text:'
RAI_M5C_PREFIX = 'tensor:mil-nce:m5c:'
RAI_VIS_PREFIX = 'tensor:mil-nce:vis:'
RAI_PSEUDO_SUM_PREFIX = 'pseudo:summary:'
RAI_PSEUDO_SCORE_PREFIX = 'pseudo:scores:'


def get_headline_key(pk: str):
    return RAI_HEADLINE_PREFIX + pk


def get_topic_key(pk: str):
    return RAI_TOPIC_PREFIX + pk


def get_text_key(pk: str):
    return RAI_TEXT_PREFIX + pk


def get_m5c_key(pk: str):
    return RAI_M5C_PREFIX + pk


def get_vis_key(pk: str):
    return RAI_VIS_PREFIX + pk


def get_sum_key(pk: str):
    return RAI_PSEUDO_SUM_PREFIX + pk


def get_score_key(pk: str):
    return RAI_PSEUDO_SCORE_PREFIX + pk
