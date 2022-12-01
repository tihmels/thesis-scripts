#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

from argparse import ArgumentParser
from pathlib import Path

from database.model import MainVideo, ShortVideo

parser = ArgumentParser('Estimate Similarity between videos')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help='Tagesschau video file(s)')
parser.add_argument('--overwrite', action='store_false', dest='skip_existing', help='')


def estimate_similarity(ts100, ts15):
    return 50


def process_video(ts15: MainVideo):
    pre_videos = ShortVideo.find((ShortVideo.suc_main.ref_pk == ts15.pk) & (ShortVideo.suc_main.temp_dist < 20000)).all()
    suc_videos = ShortVideo.find(ShortVideo.pre_main.ref_pk == ts15.pk).all()

    for ts100 in pre_videos:
        similarity = estimate_similarity(ts100, ts15)
        ts100.pre_main.similarity = similarity
        ts100.save()

    for ts100 in suc_videos:
        similarity = estimate_similarity(ts100, ts15)
        ts100.suc_main.similarity = similarity
        ts100.save()


def main():
    main_videos = list(MainVideo.all_pks())

    assert len(main_videos) > 0, 'No suitable video files have been found.'

    for idx, pk in enumerate(main_videos):
        video = MainVideo.get(pk)

        print(f'[{idx + 1}/{len(main_videos)}] {video.pk}')

        process_video(video)


if __name__ == "__main__":
    main()
