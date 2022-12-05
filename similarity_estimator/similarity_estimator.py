#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.spatial import distance
from sentence_transformers import util
from smartredis import Client

from database.model import MainVideo, ShortVideo

parser = ArgumentParser('Estimate Similarity between videos')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help='Tagesschau video file(s)')

db_address = "localhost:6379"
client = Client(address=db_address)


def visual_similarity(ts15_stories, ts100_stories):
    for ts15_story in ts15_stories:
        ts15_ai_tensors = [client.get_tensor(shot.pk) for shot in ts15_story.shots]

        for ts100_story in ts100_stories:
            ts100_ai_tensors = [client.get_tensor(shot.pk) for shot in ts100_story.shots if
                                client.tensor_exists(shot.pk)]

            if len(ts15_ai_tensors) == 0 or len(ts100_ai_tensors) == 0 or not ts15_ai_tensors[0].shape == \
                                                                              ts100_ai_tensors[0].shape:
                return

            cosine_scores = distance.cdist(ts15_ai_tensors, ts100_ai_tensors, metric='correlation')

            argmax, argmin = np.argmax(cosine_scores), np.argmin(cosine_scores)
            argmax_idx, argmin_idx = np.unravel_index(argmax, cosine_scores.shape), np.unravel_index(argmin,
                                                                                                     cosine_scores.shape)

            print(f'{ts15_story.headline} - {ts100_story.headline}')
            print(
                f'MAX[ {ts15_story.shots[argmax_idx[0]].keyframe} : {ts100_story.shots[argmax_idx[1]].keyframe} ] {np.max(cosine_scores)}')
            print(
                f'MIN[ {ts15_story.shots[argmin_idx[0]].keyframe} : {ts100_story.shots[argmin_idx[1]].keyframe} ] {np.min(cosine_scores)}')
            print(f'Mean: {np.mean(cosine_scores)}')
            print()


def sent_similarity(ts15_stories, ts100_stories):
    for ts15_story in ts15_stories:
        ts15_ai_tensors = [client.get_tensor(sentence.pk) for sentence in ts15_story.sentences]

        for ts100_story in ts100_stories:
            ts100_ai_tensors = [client.get_tensor(sentence.pk) for sentence in ts100_story.sentences if
                                client.tensor_exists(sentence.pk)]

            if len(ts15_ai_tensors) == 0 or len(ts100_ai_tensors) == 0:
                continue

            cosine_scores = util.cos_sim(ts15_ai_tensors, ts100_ai_tensors)

            pairs = []
            for i in range(cosine_scores.shape[0]):
                for j in range(i + 1, cosine_scores.shape[1]):
                    pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

            # Sort scores in decreasing order
            pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

            for pair in pairs[0:10]:
                i, j = pair['index']
                print(
                    "{} \t\t {} \t\t Score: {:.4f}".format(ts15_story.sentences[i].text, ts100_story.sentences[j].text,
                                                           pair['score']))
                print()


def process_video(ts15: MainVideo):
    videos = ShortVideo.find(
        (ShortVideo.pre_main.ref_pk == ts15.pk) or (ShortVideo.suc_main.ref_pk == ts15.pk)).sort_by('timestamp').all()

    for ts100 in videos:
        visual_similarity(ts15.stories, ts100.stories)
        # sent_similarity(ts15.stories, ts100.stories)


def main():
    videos = MainVideo.find().sort_by('timestamp').all()

    assert len(videos) > 0, 'No suitable video files have been found.'

    for idx, video in enumerate(videos):
        print(f'[{idx + 1}/{len(videos)}] {video.pk}')

        process_video(video)


if __name__ == "__main__":
    main()
