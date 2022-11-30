#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import datetime
from abc import ABC
from pathlib import Path
from typing import List, Optional

import redis
from redis_om import Field, Migrator, JsonModel

from common.VAO import get_date_time, VAO
from common.utils import frame_idx_to_time

parser = argparse.ArgumentParser('Uploads filesystem data to a Redis instance')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help="Tagesschau video file(s)")
parser.add_argument('--host', type=str, default="localhost")
parser.add_argument('--port', type=int, default=6379)
parser.add_argument('--db', type=int, default=0)
parser.add_argument('--reset', action='store_true')
args = parser.parse_args()

redis = redis.Redis(host=args.host, port=args.port, db=args.db, decode_responses=True)


# redis = redis.Redis(host=args.host, port=args.port, db=args.db, decode_responses=True, username='default', password = 'YX0Nx3ddpclPyewTGzvswBnZrPyT9Tit')

class BaseModel(JsonModel, ABC):
    class Meta:
        database = redis
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


class Shot(EmbeddedBaseModel):
    first_frame_idx: int
    last_frame_idx: int
    keyframe: str
    type: Optional[str]


class Story(EmbeddedBaseModel):
    headline: str
    first_shot: Shot
    last_shot: Shot
    duration: datetime.time
    transcript: str


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


class ShortVideo(VideoBaseModel):
    banners: List[Banner]
    is_nightly: int = Field(index=True)

    pre_main: Optional[VideoRef]
    suc_main: Optional[VideoRef]

    class Meta:
        model_key_prefix = 'ts100'


def upload_video_data(vao: VAO):
    shots = [Shot(first_frame_idx=shot.first_frame_idx,
                  last_frame_idx=shot.last_frame_idx,
                  keyframe=str(vao.data.keyframes[idx]),
                  type=shot.type) for idx, shot in enumerate(vao.data.shots)]

    transcripts = [Transcript(from_time=transcript.start,
                              to_time=transcript.end,
                              text=transcript.text) for transcript in vao.data.transcripts]

    if vao.is_summary:
        banners = [Banner(text=banner.text, confidence=banner.confidence) for banner in vao.data.banners]

        stories = [Story(headline=story.headline,
                         first_shot=shots[story.first_shot_idx],
                         last_shot=shots[story.last_shot_idx],
                         duration=frame_idx_to_time(story.last_frame_idx - story.first_frame_idx).replace(
                             microsecond=0),
                         transcript=vao.data.get_story_text(idx)) for idx, story in enumerate(vao.data.stories)]

        video = ShortVideo(pk=str(vao.id),
                           path=str(vao.path),
                           date=vao.date.date(),
                           time=vao.date.time(),
                           duration=vao.duration,
                           timestamp=vao.date.timestamp(),
                           is_nightly=vao.is_nightly_version,
                           shots=shots,
                           stories=stories,
                           transcripts=transcripts,
                           banners=banners)
    else:
        topics = [Topic(title=topic) for topic in vao.data.topics]

        stories = [MainStory(topic=topics[story.ref_idx],
                             headline=story.headline,
                             first_shot=shots[story.first_shot_idx],
                             last_shot=shots[story.last_shot_idx],
                             duration=frame_idx_to_time(story.last_frame_idx - story.first_frame_idx)
                             .replace(microsecond=0),
                             transcript=vao.data.get_story_text(idx)) for idx, story in enumerate(vao.data.stories)]

        video = MainVideo(pk=str(vao.id),
                          path=str(vao.path),
                          date=vao.date.date(),
                          time=vao.date.time(),
                          duration=vao.duration,
                          timestamp=vao.date.timestamp(),
                          shots=shots,
                          stories=stories,
                          transcripts=transcripts,
                          topics=topics)
    video.save()

    return video


def suppress(func):
    try:
        return func()
    except Exception:
        return None


def link_nodes(video, model_key_prefix):
    if model_key_prefix == 'ts100':
        predecessor = suppress(ShortVideo.find(ShortVideo.timestamp < video.timestamp).sort_by('-timestamp').first)
        successor = suppress(ShortVideo.find(ShortVideo.timestamp > video.timestamp).sort_by('timestamp').first)
    else:
        predecessor = suppress(MainVideo.find(MainVideo.timestamp < video.timestamp).sort_by('-timestamp').first)
        successor = suppress(MainVideo.find(MainVideo.timestamp > video.timestamp).sort_by('timestamp').first)

    if predecessor:
        video.pre_pk = predecessor.pk
        predecessor.suc_pk = video.pk
        predecessor.save()

    if successor:
        video.suc_pk = successor.pk
        successor.pre_pk = video.pk
        successor.save()

    video.save()


def set_refs(video):
    predecessor = suppress(MainVideo.find(MainVideo.timestamp < video.timestamp).sort_by('-timestamp').first)
    successor = suppress(MainVideo.find(MainVideo.timestamp > video.timestamp).sort_by('timestamp').first)

    if predecessor:
        video.pre_main = VideoRef(ref_pk=predecessor.pk, temp_dist=video.timestamp - predecessor.timestamp)

    if successor:
        video.suc_main = VideoRef(ref_pk=successor.pk, temp_dist=successor.timestamp - video.timestamp)

    video.save()


def main(args):
    if args.reset:
        redis.flushall()

    video_files = {file for file in args.files}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    for idx, vf in enumerate(video_files):
        vao = VAO(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vao}')

        video = upload_video_data(vao)

        video.save()

    Migrator().run()

    main_pks = MainVideo.all_pks()
    short_pks = ShortVideo.all_pks()

    for pk in main_pks:
        video = MainVideo.get(pk)
        link_nodes(video, 'ts15')

    for pk in short_pks:
        video = ShortVideo.get(pk)
        link_nodes(video, 'ts100')
        set_refs(video)

    Migrator().run()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
