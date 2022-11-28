import argparse
import datetime
from abc import ABC
from pathlib import Path
from typing import List, Optional

import redis
from redis_om import JsonModel, Field, EmbeddedJsonModel

from common.VAO import get_date_time, VAO

parser = argparse.ArgumentParser('Uploads filesystem data to a Redis instance')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help="Tagesschau video file(s)")
parser.add_argument('--host', type=str, default="localhost")
parser.add_argument('--port', type=int, default=6379)
parser.add_argument('--db', type=int, default=0)


class Transcript(EmbeddedJsonModel):
    from_time: datetime.time
    to_time: datetime.time
    text: str


class Shot(EmbeddedJsonModel):
    first_frame_idx: int
    last_frame_idx: int
    type: Optional[str]


class Topic(EmbeddedJsonModel):
    title: str


class Caption(EmbeddedJsonModel):
    text: str
    confidence: int


class Story(EmbeddedJsonModel):
    first_shot_idx: int
    last_shot_idx: int


class MainStory(Story):
    topic: Topic


class ShortStory(Story):
    caption: Caption


class VideoBaseModel(JsonModel, ABC):
    path: str = Field(index=True)
    date: datetime.datetime = Field(index=True)
    keyframes: List[str]
    shots: List[Shot]
    transcripts: List[Transcript]

    class Meta:
        global_key_prefix = 'tsv'


class MainVideo(VideoBaseModel):
    stories: List[MainStory]
    topics: List[Topic]

    class Meta:
        model_key_prefix = 'ts15'


class ShortVideo(VideoBaseModel):
    stories: List[ShortStory]
    captions: List[Caption]

    class Meta:
        model_key_prefix = 'ts100'


def upload_video_data(vao: VAO):
    shots = [Shot(first_frame_idx=shot.first_frame_idx,
                  last_frame_idx=shot.last_frame_idx,
                  type=shot.type) for shot in vao.data.shots]
    transcripts = [Transcript(from_time=transcript.start,
                              to_time=transcript.end,
                              text=transcript.text) for transcript in vao.data.transcripts]

    if vao.is_summary:
        captions = [Caption(text=caption.text, confidence=caption.confidence) for caption in vao.data.banners]
        stories = [ShortStory(first_shot_idx=story.first_shot_idx,
                              last_shot_idx=story.last_shot_idx,
                              caption=captions[story.caption_idx]) for story in vao.data.stories]

        video = ShortVideo(path=str(vao.path),
                           date=vao.date,
                           keyframes=[str(kf) for kf in vao.data.keyframes],
                           shots=shots,
                           transcripts=transcripts,
                           captions=captions,
                           stories=stories)
    else:

        video = MainVideo(path=str(vao.path),
                          date=vao.date,
                          keyframes=[str(kf) for kf in vao.data.keyframes],
                          shots=[
                              Shot(first_frame_idx=shot.first_frame_idx, last_frame_idx=shot.last_frame_idx,
                                   type=shot.type) for shot in vao.data.shots],
                          stories=[Story(title=story.title, first_shot_idx=story.first_shot_idx,
                                         last_shot_idx=story.last_shot_idx) for story in vao.data.stories],
                          transcript=[Transcript(start=transcript.start, end=transcript.end, text=transcript.text) for
                                      transcript in vao.data.transcripts],
                          topics=vao.data.topics)

    video.save()


def main(args):
    r = redis.StrictRedis(host=args.host, port=args.port, db=args.db, decode_responses=True)
    r.flushall()

    video_files = {file for file in args.files}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    for idx, vf in enumerate(video_files):
        vao = VAO(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vao}')

        upload_video_data(vao)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
