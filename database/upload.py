#!/Users/tihmels/Scripts/thesis-scripts/venv/bin/python -u
import argparse
from datetime import timedelta
from pathlib import Path

from libretranslatepy import LibreTranslateAPI
from redis_om import Migrator

from common.VAO import get_date_time, VAO, get_text
from common.utils import frame_idx_to_time, frame_idx_to_sec
from database import db
from database.model import Banner, Story, ShortVideo, ShortShot, MainShot, Transcript, MainVideo, VideoRef

parser = argparse.ArgumentParser('Uploads filesystem data to a Redis instance')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help="Tagesschau video file(s)")
parser.add_argument('--reset', action='store_true')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing')
args = parser.parse_args()

lt = LibreTranslateAPI("http://127.0.0.1:5005")


def trans(text):
    if text:
        return lt.translate(text, 'de', 'en')
    else:
        return ""


def get_story_pk(video_pk: str, story_idx: int):
    suffix = "{:02d}".format(story_idx)
    return video_pk + "-" + suffix


def create_video_data(vao: VAO, skip_existing):
    pk = str(vao.id)

    if skip_existing:
        prefix = 'tsv:ts100:' if vao.is_summary else 'tsv:ts15:'
        if db.hash_exists(prefix + pk):
            return

    transcripts = [Transcript(from_time=transcript.start,
                              to_time=transcript.end,
                              text=transcript.text.strip()) for transcript in vao.data.transcripts]

    if vao.is_summary:

        if vao.is_nightly_version:
            return

        banners = [Banner(headline=banner.headline, subheadline=banner.subline, confidence=banner.confidence)
                   for banner in vao.data.banners]

        shots = [ShortShot(first_frame_idx=shot.first_frame_idx,
                           last_frame_idx=shot.last_frame_idx,
                           duration=frame_idx_to_time(shot.last_frame_idx - shot.first_frame_idx),
                           keyframe=str(vao.data.keyframes[idx]),
                           transcript_de=get_text(vao.data.get_shot_transcripts(idx)),
                           transcript_en=trans(get_text(vao.data.get_shot_transcripts(idx))),
                           banner=banner) for idx, (shot, banner) in enumerate(zip(vao.data.shots, banners))]

        stories = [Story(pk=get_story_pk(pk, idx),
                         headline=story.headline,
                         video=str(vao.path),
                         type='ts100',
                         first_frame_idx=story.first_frame_idx,
                         last_frame_idx=story.last_frame_idx,
                         start=frame_idx_to_time(story.first_frame_idx),
                         end=frame_idx_to_time(story.last_frame_idx),
                         timestamp=(vao.date + timedelta(seconds=frame_idx_to_sec(story.first_frame_idx))).timestamp(),
                         duration=frame_idx_to_time(story.last_frame_idx - story.first_frame_idx).replace(
                             microsecond=0),
                         frames=[str(frame) for frame in vao.data.frames[story.first_frame_idx:story.last_frame_idx]],
                         shots=[shots[idx] for idx in range(story.first_shot_idx, story.last_shot_idx + 1)],
                         sentences_de=vao.data.get_story_sentences(idx),
                         sentences_en=[trans(sent) for sent in
                                       vao.data.get_story_sentences(idx)]).save() for
                   idx, story in enumerate(vao.data.stories)]

        video = ShortVideo(pk=pk,
                           path=str(vao.path),
                           date=vao.date.date(),
                           time=vao.date.time(),
                           duration=vao.duration,
                           timestamp=vao.date.timestamp(),
                           shots=shots,
                           stories=stories,
                           transcripts=transcripts)
    else:
        shots = [MainShot(first_frame_idx=shot.first_frame_idx,
                          last_frame_idx=shot.last_frame_idx,
                          duration=frame_idx_to_time(shot.last_frame_idx - shot.first_frame_idx),
                          keyframe=str(vao.data.keyframes[idx]),
                          transcript_de=get_text(vao.data.get_shot_transcripts(idx)),
                          transcript_en=trans(get_text(vao.data.get_shot_transcripts(idx))),
                          type=shot.type) for idx, shot in enumerate(vao.data.shots)]

        stories = [Story(pk=get_story_pk(pk, idx),
                         headline=vao.data.topics[story.ref_idx],
                         video=str(vao.path),
                         type='ts15',
                         first_frame_idx=story.first_frame_idx,
                         last_frame_idx=story.last_frame_idx,
                         start=frame_idx_to_time(story.first_frame_idx),
                         end=frame_idx_to_time(story.last_frame_idx),
                         timestamp=(vao.date + timedelta(seconds=frame_idx_to_sec(story.first_frame_idx))).timestamp(),
                         duration=frame_idx_to_time(story.last_frame_idx - story.first_frame_idx)
                         .replace(microsecond=0),
                         frames=[str(frame) for frame in vao.data.frames[story.first_frame_idx:story.last_frame_idx]],
                         shots=[shots[idx] for idx in range(story.first_shot_idx, story.last_shot_idx + 1)],
                         sentences_de=vao.data.get_story_sentences(idx),
                         sentences_en=[trans(sent) for sent in
                                       vao.data.get_story_sentences(idx)]).save()
                   for idx, story in enumerate(vao.data.stories)]

        video = MainVideo(pk=pk,
                          path=str(vao.path),
                          date=vao.date.date(),
                          time=vao.date.time(),
                          duration=vao.duration,
                          timestamp=vao.date.timestamp(),
                          shots=shots,
                          stories=stories,
                          transcripts=transcripts,
                          topics=vao.data.topics)

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
        db.flushall()

    video_files = {file for file in args.files}

    assert len(video_files) > 0, 'No suitable video files have been found.'

    video_files = sorted(video_files, key=get_date_time)

    for idx, vf in enumerate(video_files):
        vao = VAO(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vao}')

        video = create_video_data(vao, args.skip_existing)

        if video:
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
