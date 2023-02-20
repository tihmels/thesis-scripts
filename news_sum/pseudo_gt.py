import argparse
import math
import os
import random
import sys
from pathlib import Path
from shutil import copy

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as io
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

from common.utils import read_images, create_dir, flatten, Range
from database import rai, db
from database.model import TopicCluster, Story, get_sum_key, get_score_key, get_vis_key

parser = argparse.ArgumentParser('Pseudo Summary Generation')
parser.add_argument('--index', type=int, nargs='*', help="Generate pseudo summary for cluster index")
parser.add_argument('--fps', type=int, default=8)
parser.add_argument('--window', type=int, default=16)
parser.add_argument('--pseudo_video_dir', type=str, default='')
parser.add_argument('--save_fig', action='store_true')
parser.add_argument(
    "-th",
    "--threshold",
    default=0.8,
    type=float,
    help="cut off threshold",
)


def range_overlap(r1: Range, r2: Range):
    latest_start = max(r1.first_frame_idx, r2.first_frame_idx)
    earliest_end = min(r1.last_frame_idx, r2.last_frame_idx)
    delta = (earliest_end - latest_start)
    overlap = max(0, delta)

    return overlap


def score_per_seg(segments, scores):
    return [[seg_score] * (seg[1] - seg[0] + 1) for seg_score, seg in zip(scores, segments)]


def segment_idx_to_frame_idx(offset, segment_idx, fps=8, window=16):
    skip_n = math.floor(25 / fps)

    return offset + (segment_idx * skip_n * window)


def segment_to_frame_range(offset, first_segment_idx: int, last_segment_idx: int = None, fps=8, window=16):
    last_segment_idx = last_segment_idx + 1 if last_segment_idx else first_segment_idx + 1
    return Range(segment_idx_to_frame_idx(offset, first_segment_idx, fps, window),
                 segment_idx_to_frame_idx(offset, last_segment_idx, fps, window))


def mean_segment_similarity(segment_features, other_segment_features, mean_co=None):
    similarity_matrix = np.matmul(segment_features, other_segment_features.T)

    similarity_matrix = np.sort(similarity_matrix, axis=1)
    segment_similarity = similarity_matrix[:, -mean_co:].mean(axis=1) if mean_co else similarity_matrix.mean(axis=1)

    return segment_similarity


def get_inter_cluster_similarity_matmul(segment_features, other_clusters, mean_co=None):
    segment_similarities = []

    other_stories = flatten([cluster.ts15s for cluster in other_clusters])

    for story in other_stories:
        _, features = extract_shot_features(story, 8, 16)

        segment_similarity = mean_segment_similarity(segment_features,
                                                     np.stack(features),
                                                     mean_co=mean_co)

        segment_similarities.append(segment_similarity)

    segment_similarities = np.stack(segment_similarities, axis=1)

    return segment_similarities.mean(axis=1)


def get_inter_cluster_similarity(segment_features, other_clusters, mean_co=None):
    segment_similarities = []

    other_stories = flatten([cluster.ts15s for cluster in other_clusters])

    for story in other_stories:
        _, features = extract_shot_features(story, args.fps, args.window)

        segment_similarity = mean_segment_similarity(segment_features,
                                                     np.stack(features),
                                                     mean_co=mean_co)

        segment_similarities.append(segment_similarity)

    segment_similarities = np.stack(segment_similarities, axis=1)

    return segment_similarities.mean(axis=1)


def extract_shot_features(story: Story, fps, window):
    shot_features = []
    shot_segments = [(0, 0)]

    segment_features = torch.tensor(rai.get_tensor(get_vis_key(story.pk)))

    shots = [Range(story.first_frame_idx, story.last_frame_idx) for story in story.shots]

    curr_shot_idx = 0

    acc_segment_feature = segment_features[0]
    moving_avg_count = 1

    for seg_idx in range(1, len(segment_features)):
        segment_range = segment_to_frame_range(story.first_frame_idx, seg_idx, fps=fps, window=window)

        shot_overlaps = [range_overlap(segment_range, shot_range) for shot_range in shots]
        max_overlap = np.argmax(shot_overlaps)

        if max_overlap == curr_shot_idx:
            acc_segment_feature += segment_features[seg_idx]
            moving_avg_count += 1
        else:
            shot_segments[-1] = (shot_segments[-1][0], seg_idx - 1)
            shot_features.append(acc_segment_feature / moving_avg_count)

            shot_segments.append((seg_idx, seg_idx))

            curr_shot_idx = max_overlap
            acc_segment_feature = segment_features[seg_idx]
            moving_avg_count = 1

    if moving_avg_count > 1:
        shot_segments[-1] = (shot_segments[-1][0], len(segment_features) - 1)
        shot_features.append(acc_segment_feature / moving_avg_count)
    else:
        shot_features.append(acc_segment_feature)

    assert len(shot_features) == len(shot_segments)

    return shot_segments, shot_features


def process_cluster(cluster: TopicCluster, other_clusters: [TopicCluster], args):
    ts15_stories = [story for story in cluster.ts15s if
                    rai.tensor_exists(get_vis_key(story.pk)) and len(story.shots) > 1]
    ts100_stories = [story for story in cluster.ts100s if
                     rai.tensor_exists(get_vis_key(story.pk))]

    print(f'Keywords: {cluster.keywords[:5]}')
    print(f'{len(ts15_stories)} ts15')
    print(f'{len(ts100_stories)} ts100', end='\n\n')

    segment_features_per_story = []
    segments_per_story = []
    all_segment_features = []

    for story in ts15_stories:
        segments, features = extract_shot_features(story, args.fps, args.window)

        segment_features_per_story.append(torch.stack(features))
        segments_per_story.append(segments)
        all_segment_features.extend(features)

    all_segment_features = torch.stack(all_segment_features)

    ts100_segment_features = []

    for story in ts100_stories:
        _, features = extract_shot_features(story, args.fps, args.window)

        ts100_segment_features.extend(features)

    ts100_segment_features = torch.stack(ts100_segment_features)

    all_other_features = []

    other_stories = flatten([cluster.ts15s for cluster in other_clusters])
    for story in other_stories:
        _, features = extract_shot_features(story, args.fps, args.window)

        all_other_features.extend(features)

    all_other_features = torch.stack(all_other_features)

    for idx, (shot_segments, shot_features) in enumerate(zip(segments_per_story, segment_features_per_story)):

        story = ts15_stories[idx]

        print(
            f"[{idx + 1}/{len(cluster.ts15s)}] Story: {story.headline} "
            f"({story.pk}) {story.video} {story.start} - {story.end}")

        inter_cluster_sim = mean_segment_similarity(shot_features, all_other_features)
        intra_cluster_sim = mean_segment_similarity(shot_features, all_segment_features)
        ts100_sim = mean_segment_similarity(shot_features, ts100_segment_features)

        topic_relevance_score = intra_cluster_sim - inter_cluster_sim

        segment_scores = topic_relevance_score + ts100_sim
        segment_scores = F.normalize(torch.Tensor(segment_scores), dim=0).numpy()
        segment_scores[0] = min(segment_scores)

        threshold = args.threshold * segment_scores.max()
        n_segments = shot_segments[-1][1] + 1

        if args.save_fig:
            plot_it(shot_segments,
                    threshold,
                    inter_cluster_sim,
                    intra_cluster_sim,
                    ts100_sim,
                    segment_scores)

            plt.title(label=f"{story.pk}", fontdict={'fontsize': 14})

            save_path = f'/Users/tihmels/Desktop/pseudogen/cluster/{cluster.index}/'
            create_dir(Path(save_path))

            plt.savefig(f'{save_path}/{story.pk}.jpg', bbox_inches=0)
            plt.close('all')

        machine_summary = np.zeros(n_segments)
        machine_summary_scores = np.zeros(n_segments)

        random_video_bool = random.random() < 0.25 and args.pseudo_video_dir

        summary_video = []

        for segment, score in zip(shot_segments, segment_scores):
            start, end = segment

            if end >= start:
                machine_summary_scores[start: end + 1] = score
                if score >= threshold:
                    machine_summary[start: end + 1] = 1

                    if random_video_bool:
                        from_idx, to_idx = segment_to_frame_range(0, start, end)
                        frames = story.frames[from_idx:to_idx]
                        summary_video.append(torch.tensor(read_images(frames)))

        if random_video_bool and len(summary_video) > 0:
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

        redis_summary.clear()
        redis_scores.clear()

        redis_summary.extend(machine_summary.tolist())
        redis_scores.extend(machine_summary_scores.tolist())


def copy_keyframes(shot_segments, story, path="/Users/tihmels/Desktop/keyframes/"):
    keyframe_folder = Path(path)
    create_dir(keyframe_folder, rm_if_exist=True)
    for seg_idx, segment in enumerate(shot_segments):
        frame_range = segment_to_frame_range(0, segment[0], segment[1], fps=8, window=16)
        seg_frame = story.frames[int((frame_range[0] + frame_range[1]) / 2)]

        copy(seg_frame, Path(keyframe_folder, f'S{seg_idx}.jpg'))


def plot_it(segments,
            threshold,
            inter_cluster_sim,
            intra_cluster_sim,
            ts100_sim,
            final_score):
    n_segments = segments[-1][1] + 1
    x = list(range(n_segments))

    fig = plt.figure(1, figsize=(18, 4))
    plt.subplots_adjust(bottom=0.18, left=0.05, right=0.98)
    ax = fig.add_subplot(111)

    plt.xlabel('Segment')
    plt.ylabel('Score')

    y_inter = flatten(score_per_seg(segments, inter_cluster_sim))
    y_intra = flatten(score_per_seg(segments, intra_cluster_sim))
    y_ts100 = flatten(score_per_seg(segments, ts100_sim))
    y_final = flatten(score_per_seg(segments, final_score))

    y_min = min(min(y_inter),
                min(y_intra),
                min(y_ts100),
                min(y_final),
                threshold)

    y_max = max(max(y_inter),
                max(y_intra),
                max(y_ts100),
                max(y_final),
                threshold)

    y_ax_min = y_min - (y_max - y_min) * 0.1
    y_ax_max = y_max + (y_max - y_min) * 0.1

    ax.axis([0, n_segments - 1, y_ax_min, y_ax_max])

    ax.hlines(y=threshold, xmin=0, xmax=n_segments, linewidth=0.5, color=(0, 0, 0, 1))
    ax.vlines(flatten(segments), y_ax_min, y_ax_max, linestyles='dotted', colors='grey')

    ax.plot(x, y_inter, color='r', linewidth=0.5, label='Inter-cluster similarity')
    ax.plot(x, y_intra, color='g', linewidth=0.5, label='Intra-cluster similarity')
    ax.plot(x, y_ts100, color='c', linewidth=0.5, label='ts100 similarity')
    ax.plot(x, y_final, color='b', linewidth=0.8, label="Final score")

    ax.legend()
    plt.xticks(range(0, n_segments, 5))
    ax.fill_between(x, y_ax_min, y_ax_max, where=(y_final > threshold), color='b', alpha=.1)

    return ax


def main(args):
    if args.index:
        clusters = [TopicCluster.find(TopicCluster.index == index).first() for index in args.index]
    else:
        clusters = TopicCluster.find().all()
        random.shuffle(clusters)

    for cluster in clusters:
        print(f'----- Cluster {cluster.index} -----')

        other_clusters = TopicCluster.find(TopicCluster.index != cluster.index).all()

        process_cluster(cluster, other_clusters, args)

        print()

    sys.exit()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
