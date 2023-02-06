import argparse
import matplotlib
import numpy as np
import os
import random
import sys
import torch
import torchvision.io as io
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.use('TkAgg')

from common.utils import read_images, create_dir, flatten
from database import rai, db
from database.model import TopicCluster, Story, get_text_key, get_vis_key, get_sum_key, get_score_key

parser = argparse.ArgumentParser('Pseudo Summary Generation')
parser.add_argument('--index', type=int, nargs='*', help="Generate pseudo summary for cluster index")
parser.add_argument('--pseudo_video_dir', type=str, default='')

sim_thresh = 1


def score_per_seg(segments, scores):
    return [[seg_score] * (seg[1] - seg[0] + 1) for seg_score, seg in zip(scores, segments)]


def segment_idx_to_frame_idx(story_start_idx, segment_idx):
    return story_start_idx + (segment_idx * 2 * 32)


def segment_to_frame_range(story_start_idx, first_segment_idx: int, last_segment_idx: int = None):
    last_segment_idx = last_segment_idx + 1 if last_segment_idx else first_segment_idx + 1
    return (segment_idx_to_frame_idx(story_start_idx, first_segment_idx),
            segment_idx_to_frame_idx(story_start_idx, last_segment_idx))


def mean_segment_similarity(segment_features, other_segment_features, mean_co=None):
    segment_similarity = cosine_similarity(segment_features, other_segment_features)

    segment_similarity = np.sort(segment_similarity, axis=1)
    segment_similarity = segment_similarity[:, -mean_co:].mean(axis=1) if mean_co else segment_similarity.mean(axis=1)

    return segment_similarity


def mean_segment_similarity_matmul(segment_features, other_segment_features, mean_co=None):
    segment_similarity = np.matmul(segment_features, other_segment_features.T)

    segment_similarity = np.sort(segment_similarity, axis=1)
    segment_similarity = segment_similarity[:, -mean_co:].mean(axis=1) if mean_co else segment_similarity.mean(axis=1)

    return segment_similarity


def get_inter_cluster_similarity_matmul(segment_features, other_clusters, mean_co=None):
    segment_similarities = []

    other_stories = flatten([cluster.ts15s for cluster in other_clusters])

    for story in other_stories:
        _, features = extract_segment_features(story, sim_thresh=sim_thresh)

        segment_similarity = mean_segment_similarity_matmul(segment_features,
                                                            np.stack(features),
                                                            mean_co=mean_co)

        segment_similarities.append(segment_similarity)

    segment_similarities = np.stack(segment_similarities, axis=1)

    return segment_similarities.mean(axis=1)


def get_inter_cluster_similarity(segment_features, other_clusters, mean_co=None):
    segment_similarities = []

    other_stories = flatten([cluster.ts15s for cluster in other_clusters])

    for story in other_stories:
        _, features = extract_segment_features(story, sim_thresh=sim_thresh)

        segment_similarity = mean_segment_similarity(segment_features,
                                                     np.stack(features),
                                                     mean_co=mean_co)

        segment_similarities.append(segment_similarity)

    segment_similarities = np.stack(segment_similarities, axis=1)

    return segment_similarities.mean(axis=1)


def get_text_similarity(segment_features, story_pk):
    if rai.tensor_exists(get_text_key(story_pk)):
        text_features = torch.tensor(rai.get_tensor(get_text_key(story_pk)))

        text_similarity = mean_segment_similarity(segment_features, text_features, mean_co=1)

        return text_similarity

    return None


def extract_segment_features(story: Story, sim_thresh=0.95):
    story_segment_features = []
    story_segments = [(0, 0)]  # story segments (from, to) - inclusive to
    story_segment_count = 1

    segment_features = torch.tensor(rai.get_tensor(get_vis_key(story.pk)))

    similarity_matrix = torch.matmul(segment_features, segment_features.t())
    similarity_means = similarity_matrix.mean(axis=1)

    max_similarity = similarity_means.max()

    ref_segment_feature = segment_features[0]
    moving_avg_count = 1

    for seg_idx in range(1, len(segment_features)):
        curr_segment = segment_features[seg_idx]
        similarity = torch.matmul(curr_segment, ref_segment_feature.t())

        if similarity > sim_thresh * max_similarity:
            ref_segment_feature = torch.div(ref_segment_feature + curr_segment, 2)

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


def align_mean(population, mean):
    pop_mean = np.mean(population)
    if pop_mean > mean:
        return population - (pop_mean - mean)
    else:
        return population + (mean - pop_mean)


def process_cluster(cluster: TopicCluster, other_clusters: [TopicCluster]):
    ts15_stories = [story for story in cluster.ts15s if
                    rai.tensor_exists(get_vis_key(story.pk)) and len(story.shots) > 1]
    ts100_stories = [story for story in cluster.ts100s if
                     rai.tensor_exists(get_vis_key(story.pk)) and story.is_nightly == 0]

    print(f'Keywords: {cluster.keywords[:5]}')
    print(f'{len(ts15_stories)} ts15')
    print(f'{len(ts100_stories)} ts100', end='\n\n')

    segment_features_per_story = []
    segments_per_story = []
    all_segment_features = []

    for story in ts15_stories:
        segments, features = extract_segment_features(story, sim_thresh=sim_thresh)

        segment_features_per_story.append(torch.stack(features))
        segments_per_story.append(segments)
        all_segment_features.extend(features)

    all_segment_features = torch.stack(all_segment_features)

    ts100_segment_features = []

    for story in ts100_stories:
        _, features = extract_segment_features(story, sim_thresh=sim_thresh)

        ts100_segment_features.extend(features)

    ts100_segment_features = torch.stack(ts100_segment_features)

    all_other_features = []

    other_stories = flatten([cluster.ts15s for cluster in other_clusters])
    for story in other_stories:
        _, features = extract_segment_features(story, sim_thresh=sim_thresh)

        all_other_features.extend(features)

    all_other_features = torch.stack(all_other_features)
    all_features = torch.cat((all_segment_features, all_other_features))

    inter_cluster_threshold = cosine_similarity(all_features, all_features).mean(axis=1).max()
    intra_cluster_threshold = cosine_similarity(all_segment_features, all_segment_features).mean(axis=1).max()

    threshold = (inter_cluster_threshold + intra_cluster_threshold) / 2

    for idx, (segments, segment_features) in enumerate(zip(segments_per_story, segment_features_per_story)):

        story = ts15_stories[idx]

        print(
            f"[{idx + 1}/{len(cluster.ts15s)}] Story: {story.headline} "
            f"({story.pk}) {story.video} {story.start} - {story.end}")

        inter_cluster_sim = get_inter_cluster_similarity(segment_features, other_clusters, mean_co=1)
        inter_cluster_dist = 1 - inter_cluster_sim

        inter_mean = np.mean(inter_cluster_dist)

        intra_cluster_sim = mean_segment_similarity(segment_features, all_segment_features, mean_co=5)
        intra_cluster_dist = align_mean(1 - intra_cluster_sim, inter_mean)

        ts100_sim = mean_segment_similarity(segment_features, ts100_segment_features, mean_co=5)
        ts100_sim = align_mean(ts100_sim, inter_mean)

        text_sim = get_text_similarity(segment_features, story.pk)
        text_sim = align_mean(text_sim, inter_mean)

        if text_sim is not None:
            segment_scores = (inter_cluster_dist + intra_cluster_dist + ts100_sim + text_sim) / 4
        else:
            segment_scores = (inter_cluster_dist + intra_cluster_dist + ts100_sim) / 3

        n_video_segments = segments[-1][1] + 1

        mean_thresh = (threshold + np.mean(segment_scores)) / 2

        ax = plot_it(segments,
                     mean_thresh,
                     inter_cluster_dist,
                     intra_cluster_dist,
                     ts100_sim,
                     text_sim,
                     segment_scores)

        ax.hlines(y=threshold, xmin=0, xmax=n_video_segments, linewidth=0.5, linestyles='dotted', color=(0, 0, 0, 1))
        ax.hlines(y=np.mean(segment_scores), xmin=0, xmax=n_video_segments, linewidth=0.5, linestyles='dotted',
                  color=(0, 0, 0, 1))

        plt.show()

        machine_summary = np.zeros(n_video_segments)
        machine_summary_scores = np.zeros(n_video_segments)

        random_video_bool = random.random() < 0.25 and args.pseudo_video_dir

        summary_video = []

        for segment, score in zip(segments, segment_scores):
            start, end = segment

            if end >= start:
                machine_summary_scores[start: end + 1] = score
                if score >= inter_cluster_threshold:
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

        redis_summary.extend(machine_summary.tolist())
        redis_scores.extend(machine_summary_scores.tolist())


def plot_it(segments,
            threshold,
            inter_cluster_dist,
            intra_cluster_dist,
            ts100_sim,
            text_sim,
            final_score):
    n_video_segments = segments[-1][1] + 1
    x = list(range(n_video_segments))

    fig = plt.figure(1, figsize=(18, 4))
    plt.subplots_adjust(bottom=0.18, left=0.05, right=0.98)
    ax = fig.add_subplot(111)

    plt.xlabel('Segment')
    plt.ylabel('Score')

    y_inter = flatten(score_per_seg(segments, inter_cluster_dist))
    y_intra = flatten(score_per_seg(segments, intra_cluster_dist))
    y_ts100 = flatten(score_per_seg(segments, ts100_sim))
    y_text = flatten(score_per_seg(segments, text_sim))
    y_final = flatten(score_per_seg(segments, final_score))

    y_min = min(min(y_inter),
                min(y_intra),
                min(y_ts100),
                min(y_text),
                min(y_final),
                threshold)

    y_max = max(max(y_inter),
                max(y_intra),
                max(y_ts100),
                max(y_text),
                max(y_final),
                threshold)

    y_ax_min = y_min - (y_max - y_min) * 0.1
    y_ax_max = y_max + (y_max - y_min) * 0.1

    ax.axis([0, n_video_segments - 1, y_ax_min, y_ax_max])

    ax.hlines(y=threshold, xmin=0, xmax=n_video_segments, linewidth=0.5, color=(0, 0, 0, 1))
    ax.vlines(flatten(segments), y_ax_min, y_ax_max, linestyles='dotted', colors='grey')

    ax.plot(x, y_inter, color='c', linewidth=0.5, label='Inter-Cluster Distance')
    ax.plot(x, y_intra, color='r', linewidth=0.5, label='Intra-Cluster Distance')
    ax.plot(x, y_ts100, color='g', linewidth=0.5, label='ts100 Similarity')
    ax.plot(x, y_text, color='y', linewidth=0.5, label='Text Similarity')
    ax.plot(x, y_final, color='b', linewidth=0.8, label="Final Score")

    ax.legend()
    plt.xticks(range(0, n_video_segments, 5))
    ax.fill_between(x, y_ax_min, y_ax_max, where=(y_final > threshold), color='b', alpha=.1)

    return ax


def main(args):
    if args.index:
        clusters = [TopicCluster.find(TopicCluster.index == index).first() for index in args.index]
    else:
        clusters = TopicCluster.find().sort_by('-index').all()
        random.shuffle(clusters)

    for cluster in clusters:
        print(f'----- Cluster {cluster.index} -----')

        other_clusters = TopicCluster.find(TopicCluster.index != cluster.index).all()

        process_cluster(cluster, other_clusters)

        print()

    sys.exit()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
