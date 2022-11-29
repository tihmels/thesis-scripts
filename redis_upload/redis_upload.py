#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import datetime
from abc import ABC
from pathlib import Path
from typing import List, Optional

import redis
from redis_om import JsonModel, Field, EmbeddedJsonModel, Migrator

from common.VAO import get_date_time, VAO
from common.utils import frame_idx_to_time

parser = argparse.ArgumentParser('Uploads filesystem data to a Redis instance')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help="Tagesschau video file(s)")
parser.add_argument('--host', type=str, default="localhost")
parser.add_argument('--port', type=int, default=6379)
parser.add_argument('--db', type=int, default=0)
args = parser.parse_args()

redis = redis.Redis(host=args.host, port=args.port, db=args.db, decode_responses=True)


class Topic(EmbeddedJsonModel):
    title: str

    class Meta:
        database = redis


class Banner(EmbeddedJsonModel):
    text: str
    confidence: int

    class Meta:
        database = redis


class Shot(EmbeddedJsonModel):
    first_frame_idx: int
    last_frame_idx: int
    keyframe: str
    type: Optional[str]

    class Meta:
        database = redis


class Story(EmbeddedJsonModel):
    headline: str
    first_shot: Shot
    last_shot: Shot
    duration: datetime.time
    transcript: str

    class Meta:
        database = redis


class MainStory(Story):
    topic: Topic


class Transcript(EmbeddedJsonModel):
    from_time: datetime.time
    to_time: datetime.time
    text: str

    class Meta:
        database = redis


class VideoBaseModel(JsonModel, ABC):
    path: str
    date: datetime.date
    time: datetime.time
    duration: datetime.time
    timestamp: int = Field(index=True, sortable=True)
    shots: List[Shot]
    stories: List[Story]
    transcripts: List[Transcript]

    class Meta:
        database = redis
        orm_mode = True
        arbitrary_types_allowed = True
        extra = "allow"
        global_key_prefix = 'tsv'


class MainVideo(VideoBaseModel):
    topics: List[Topic]

    class Meta:
        model_key_prefix = 'ts15'


class ShortVideo(VideoBaseModel):
    banners: List[Banner]
    is_nightly: int

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
                             duration=frame_idx_to_time(story.last_frame_idx - story.first_frame_idx).replace(
                                 microsecond=0),
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


def main(args):
    video_files = {file for file in args.files}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    for idx, vf in enumerate(video_files):
        vao = VAO(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vao}')

        upload_video_data(vao)

    Migrator().run()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
