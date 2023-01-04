import argparse
import matplotlib
import numpy as np
import os
import random
import sys
import torch
import torch.nn.functional as F
import torchvision.io as io
from alive_progress import alive_bar
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.use('TkAgg')

from common.utils import read_images, create_dir, flatten
from database import rai, db
from database.config import get_vis_key, get_sum_key, get_score_key, get_text_key
from database.model import TopicCluster, Story

parser = argparse.ArgumentParser('Pseudo Summary Generation')
parser.add_argument('--index', type=int, nargs='*', help="Generate pseudo summary for cluster index")
parser.add_argument('--pseudo_video_dir', type=str, default='')


def seg_scores(segments, scores):
    return [[seg_score] * (seg[1] - seg[0] + 1) for seg_score, seg in zip(scores, segments)]


def segment_idx_to_frame_idx(story_start_idx, segment_idx):
    return story_start_idx + (segment_idx * 2 * 32)


def segment_to_frame_range(story_start_idx, first_segment_idx: int, last_segment_idx: int = None):
    last_segment_idx = last_segment_idx + 1 if last_segment_idx else first_segment_idx + 1
    return (segment_idx_to_frame_idx(story_start_idx, first_segment_idx),
            segment_idx_to_frame_idx(story_start_idx, last_segment_idx))


def get_segment_similarity(segment_features, other_segment_features, cossim=True, mean_co=None):
    if cossim:
        segment_similarity = cosine_similarity(segment_features, other_segment_features)
    else:
        segment_similarity = np.matmul(segment_features, other_segment_features.T)

    segment_similarity = np.sort(segment_similarity, axis=1)
    segment_similarity = segment_similarity[:, -mean_co:].mean(axis=1) if mean_co else segment_similarity.mean(axis=1)

    return segment_similarity


def get_inter_cluster_similarity(segment_features, other_clusters, cossim, mean_co=None):
    segment_similarities = []

    for cluster in other_clusters:
        other_stories = [story for story in cluster.ts15s if rai.tensor_exists(get_vis_key(story.pk))]

        for story in other_stories:
            _, features = extract_segment_features(story)

            segment_similarity = get_segment_similarity(segment_features, np.stack(features), cossim=cossim,
                                                        mean_co=mean_co)

            segment_similarities.append(segment_similarity)

    segment_similarities = np.stack(segment_similarities, axis=1)

    return segment_similarities.mean(axis=1)


def get_text_similarity_matrix(segment_features, story_pk):
    if rai.tensor_exists(get_text_key(story_pk)):
        text_features = torch.tensor(rai.get_tensor(get_text_key(story_pk)))
        text_features = F.normalize(text_features, dim=1)

        text_similarity_matrix = np.matmul(segment_features, text_features.t())
        # text_similarity_matrix = cosine_similarity(segment_features, text_features)

        return text_similarity_matrix.numpy()

    return None


def extract_segment_features(story: Story, sim_thresh=0.95):
    story_segment_features = []
    story_segments = [(0, 0)]  # story segments (from, to) - inclusive to
    story_segment_count = 1

    segment_features = torch.tensor(rai.get_tensor(get_vis_key(story.pk)))
    segment_features = F.normalize(segment_features, dim=1)

    similarity_matrix = torch.matmul(segment_features, segment_features.t())
    similarity_means = similarity_matrix.mean(axis=1)

    max_similarity = similarity_means.max()

    ref_segment_feature = segment_features[0]
    moving_avg_count = 1

    for seg_idx in range(1, len(segment_features)):
        curr_segment = segment_features[seg_idx]
        similarity = torch.matmul(curr_segment, ref_segment_feature.t())

        if similarity > sim_thresh * max_similarity:
            ref_segment_feature = (ref_segment_feature + curr_segment) / 2

            moving_avg_count += 1
        else:
            story_segment_count += 1

            story_segments[-1] = (story_segments[-1][0], seg_idx - 1)
            story_segment_features.append(ref_segment_feature)

            story_segments.append((seg_idx, seg_idx))

            ref_segment_feature = curr_segment

            moving_avg_count = 1

    story_segments[-1] = (story_segments[-1][0], len(segment_features) - 1)
    story_segment_features.append(ref_segment_feature)

    assert story_segment_count == len(story_segments) == len(story_segment_features)

    return story_segments, story_segment_features


def process_cluster(cluster: TopicCluster, other_clusters: [TopicCluster], args):
    ts15_stories = [story for story in cluster.ts15s if rai.tensor_exists(get_vis_key(story.pk))]
    ts100_stories = [story for story in cluster.ts100s if rai.tensor_exists(get_vis_key(story.pk))]

    print(f'Keywords: {cluster.keywords[:5]}')
    print(f'{len(ts15_stories)} ts15')
    print(f'{len(ts100_stories)} ts100', end='\n\n')

    segment_features_per_story = []
    segments_per_story = []
    all_segment_features = []

    with alive_bar(len(ts15_stories),
                   ctrl_c=False, title="Feature Extraction [ts15]",
                   length=25, force_tty=True, dual_line=True, receipt_text=True) as bar:

        for story in ts15_stories:
            bar.text = f'Story: {story.pk} -> {story.video}'

            segments, features = extract_segment_features(story, sim_thresh=0.9)

            segment_features_per_story.append(torch.stack(features))
            segments_per_story.append(segments)
            all_segment_features.extend(features)

            bar()

    all_segment_features = torch.stack(all_segment_features)

    # TS100

    ts100_segment_features = []

    with alive_bar(len(ts100_stories),
                   ctrl_c=False, title="Feature Extraction [ts100]",
                   length=25, force_tty=True, dual_line=True, receipt_text=True) as bar:
        for story in ts100_stories:
            bar.text = f'Story: {story.pk} -> {story.video}'

            _, features = extract_segment_features(story, sim_thresh=0.9)

            ts100_segment_features.extend(features)
            bar()

    ts100_segment_features = torch.stack(ts100_segment_features)

    for idx, (segments, segment_features) in enumerate(zip(segments_per_story, segment_features_per_story)):

        story = ts15_stories[idx]

        print(f"[{idx + 1}/{len(cluster.ts15s)}] Story: {story.headline} ({story.pk})")

        if len(segment_features) == 0:
            print('No features could be extracted ...')
            continue

        inter_cluster_sim_matmul = get_inter_cluster_similarity(segment_features,
                                                                other_clusters,
                                                                cossim=False,
                                                                mean_co=5)
        intra_cluster_sim_matmul = get_segment_similarity(segment_features,
                                                          all_segment_features,
                                                          cossim=False,
                                                          mean_co=5)

        inter_cluster_sim_inv = 1 - inter_cluster_sim_matmul
        intra_cluster_sim_inv = 1 - intra_cluster_sim_matmul

        ts100_similarity = get_segment_similarity(segment_features,
                                                  ts100_segment_features,
                                                  cossim=False,
                                                  mean_co=5)

        text_similarity_matrix = get_text_similarity_matrix(segment_features, story.pk)

        if text_similarity_matrix is not None:
            text_similarity = text_similarity_matrix.mean(axis=1)

            segment_scores = (inter_cluster_sim_inv +
                              intra_cluster_sim_inv +
                              ts100_similarity +
                              text_similarity) / 4
        else:
            segment_scores = (inter_cluster_sim_inv +
                              intra_cluster_sim_inv +
                              ts100_similarity) / 3

        segment_scores = torch.tensor((inter_cluster_sim_inv + intra_cluster_sim_inv + ts100_similarity) / 3)
        # segment_scores = F.normalize(segment_scores, dim=0)

        n_video_segments = segments[-1][1] + 1

        threshold = segment_scores.max() * 0.85

        fig = plt.figure(1, figsize=(18, 4))
        plt.subplots_adjust(bottom=0.18, left=0.05, right=0.98)

        ax = fig.add_subplot(111)

        plt.xlabel('Segment')
        plt.ylabel('Score')

        x = list(range(n_video_segments))

        y_inter_sim = flatten(seg_scores(segments, inter_cluster_sim_matmul))
        y_inter_sim_inv = flatten(seg_scores(segments, inter_cluster_sim_inv))
        y_intra_sim = flatten(seg_scores(segments, intra_cluster_sim_matmul))
        y_intra_sim_inv = flatten(seg_scores(segments, intra_cluster_sim_inv))
        y_ts100_sim = flatten(seg_scores(segments, ts100_similarity))
        y_text_sim = flatten(seg_scores(segments, text_similarity))

        y_final = flatten(seg_scores(segments, segment_scores))

        y_min = min(min(inter_cluster_sim_inv),
                    min(intra_cluster_sim_inv),
                    max(ts100_similarity),
                    max(text_similarity))
        y_max = max(max(inter_cluster_sim_inv),
                    max(intra_cluster_sim_inv),
                    max(ts100_similarity),
                    max(text_similarity))

        y_axis_min = y_min - (y_max - y_min) * 0.1
        y_axis_max = y_max + (y_max - y_min) * 0.1

        ax.axis([0, n_video_segments - 1, y_axis_min, y_axis_max])

        ax.hlines(y=threshold, xmin=0, xmax=n_video_segments, linewidth=0.5, color=(0, 0, 0, 1))
        ax.vlines(flatten(segments), y_axis_min, y_axis_max, linestyles='dotted', colors='grey')

        ax.plot(x, y_final, color='b', linewidth=0.8, label="Final Score")
        # ax.plot(x, y_inter_sim, color='r', linewidth=0.5, label="Inter-Cluster Score")
        ax.plot(x, y_inter_sim_inv, color='g', linewidth=0.5, label="Inter-Cluster Score (inv)")
        ax.plot(x, y_intra_sim_inv, color='r', linewidth=0.5, label="Intra-Cluster Score (inv)")
        # ax.plot(x, y_ts15, color='r', linestyle='dashed', linewidth=0.5)
        # ax.plot(x, y_ts100, color='g', linewidth=0.5, label="ts100 score")
        # ax.plot(x, y_text, color='c', linewidth=0.5, label="Text score")

        ax.legend()

        plt.xticks(range(0, n_video_segments, 5))

        ax.fill_between(x, y_axis_min, y_axis_max, where=(y_final > threshold.numpy()), color='b', alpha=.1)

        machine_summary = np.zeros(n_video_segments)
        machine_summary_scores = np.zeros(n_video_segments)

        store_video_bool = random.random() < 0.2 and args.pseudo_video_dir

        summary_video = []

        for segment, score in zip(segments, segment_scores):
            start, end = segment

            if end >= start:
                machine_summary_scores[start: end + 1] = score
                if score >= threshold:
                    machine_summary[start: end + 1] = 1
                    if store_video_bool:
                        from_idx, to_idx = segment_to_frame_range(0, start, end)
                        frames = story.frames[from_idx:to_idx]
                        summary_video.append(torch.tensor(np.array(read_images(frames))))

        # plt.show()

        if store_video_bool:
            summary_video = torch.cat(summary_video, dim=0)

            pseudo_video_dir = Path(args.pseudo_video_dir, str(cluster.index))

            create_dir(pseudo_video_dir)

            io.write_video(
                os.path.join(str(pseudo_video_dir), "{}.mp4".format(story.pk)),
                summary_video,
                25,
            )

        redis_summary = db.List(get_sum_key(story.pk))
        redis_scores = db.List(get_score_key(story.pk))

        redis_summary.extend(machine_summary.tolist())
        redis_scores.extend(machine_summary_scores.tolist())


def main(args):
    if args.index:
        clusters = [TopicCluster.find(TopicCluster.index == index).first() for index in args.index]
    else:
        clusters = TopicCluster.find().sort_by('-index').all()
        random.shuffle(clusters)

    for cluster in clusters:
        print(f'----- Cluster {cluster.index} -----')

        process_cluster(cluster, TopicCluster.find(TopicCluster.index != cluster.index).sort_by('index').all(), args)
        print()

    sys.exit()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
