#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
from pathlib import Path

from redis_om import Migrator

from common.VAO import get_date_time, VAO
from common.utils import frame_idx_to_time
from database.model import Shot, Banner, Story, ShortVideo, Topic, MainStory, MainVideo, VideoRef, red, \
    Sentence, Transcript

parser = argparse.ArgumentParser('Uploads filesystem data to a Redis instance')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help="Tagesschau video file(s)")
parser.add_argument('--reset', action='store_true')
args = parser.parse_args()


def upload_video_data(vao: VAO):
    transcripts = [Transcript(from_time=transcript.start,
                              to_time=transcript.end,
                              text=transcript.text) for transcript in vao.data.transcripts]

    if vao.is_summary:
        banners = [Banner(text=banner.text, confidence=banner.confidence) for banner in vao.data.banners]

        shots = [Shot(first_frame_idx=shot.first_frame_idx,
                      last_frame_idx=shot.last_frame_idx,
                      keyframe=str(vao.data.keyframes[idx]),
                      banner=banner) for idx, (shot, banner) in enumerate(zip(vao.data.shots, banners))]

        stories = [Story(headline=story.headline,
                         first_shot=shots[story.first_shot_idx],
                         last_shot=shots[story.last_shot_idx],
                         duration=frame_idx_to_time(story.last_frame_idx - story.first_frame_idx).replace(
                             microsecond=0),
                         sentences=[Sentence(text=sent) for sent in vao.data.get_story_sentences(idx)]) for idx, story
                   in
                   enumerate(vao.data.stories)]

        video = ShortVideo(pk=str(vao.id),
                           path=str(vao.path),
                           date=vao.date.date(),
                           time=vao.date.time(),
                           duration=vao.duration,
                           timestamp=vao.date.timestamp(),
                           is_nightly=vao.is_nightly_version,
                           shots=shots,
                           stories=stories,
                           transcripts=transcripts)
    else:
        topics = [Topic(title=topic) for topic in vao.data.topics]

        shots = [Shot(first_frame_idx=shot.first_frame_idx,
                      last_frame_idx=shot.last_frame_idx,
                      keyframe=str(vao.data.keyframes[idx]),
                      type=shot.type) for idx, shot in enumerate(vao.data.shots)]

        stories = [MainStory(topic=topics[story.ref_idx],
                             headline=story.headline,
                             first_shot=shots[story.first_shot_idx],
                             last_shot=shots[story.last_shot_idx],
                             duration=frame_idx_to_time(story.last_frame_idx - story.first_frame_idx)
                             .replace(microsecond=0),
                             sentences=[Sentence(text=sent) for sent in vao.data.get_story_sentences(idx)])
                   for idx, story in enumerate(vao.data.stories)]

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


def link_nodes(video):
    if video.Meta.model_key_prefix == 'ts100':
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
        red.flushall()

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
        link_nodes(video)

    for pk in short_pks:
        video = ShortVideo.get(pk)
        link_nodes(video)
        set_refs(video)

    Migrator().run()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
