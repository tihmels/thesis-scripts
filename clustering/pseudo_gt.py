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

matplotlib.use('TkAgg')

from common.utils import read_images, create_dir, flatten
from database import rai, db
from database.config import RAI_TEXT_PREFIX, get_vis_key
from database.model import TopicCluster, Story

parser = argparse.ArgumentParser('Pseudo Summary Generation')
parser.add_argument('--index', type=int, nargs='*', help="Generate pseudo summary for cluster index")
parser.add_argument('--pseudo_video_dir', type=str, default='')


def segment_idx_to_frame_idx(story_start_idx, segment_idx):
    return story_start_idx + (segment_idx * 2 * 32)


def segment_to_frame_range(story_start_idx, first_segment_idx: int, last_segment_idx: int = None):
    last_segment_idx = last_segment_idx if last_segment_idx else first_segment_idx + 1
    return (segment_idx_to_frame_idx(story_start_idx, first_segment_idx),
            segment_idx_to_frame_idx(story_start_idx, last_segment_idx))


def get_text_similarity_matrix(story: Story, segment_features):
    if rai.tensor_exists(RAI_TEXT_PREFIX + story.pk):
        text_features = torch.tensor(rai.get_tensor(RAI_TEXT_PREFIX + story.pk))
        text_features = F.normalize(text_features, dim=1)

        text_similarity_matrix = torch.matmul(segment_features, text_features.t())
        return text_similarity_matrix

    return None


def extract_segment_features(story: Story):
    story_segment_features = []
    story_segment_count = 1
    story_segments = [(0, 0)]

    visual_segment_features = torch.tensor(rai.get_tensor(get_vis_key(story.pk)))
    visual_segment_features = F.normalize(visual_segment_features, dim=1)

    similarity_matrix = torch.matmul(visual_segment_features, visual_segment_features.t())
    similarity_means = similarity_matrix.mean(axis=1)
    max_similarity = similarity_means.max()

    # sns.heatmap(similarity_matrix, square=True)
    # plt.show()

    segment_feature = visual_segment_features[0]
    moving_avg_count = 1

    for seg_idx in range(1, len(visual_segment_features)):
        similarity = torch.matmul(visual_segment_features[seg_idx], segment_feature.t())

        if similarity > 0.85 * max_similarity:
            moving_avg_count += 1
            segment_feature = (segment_feature + visual_segment_features[seg_idx]) / 2
        else:
            story_segment_count += 1

            story_segments[-1] = (story_segments[-1][0], seg_idx - 1)
            story_segments.append((seg_idx, seg_idx))

            story_segment_features.append(segment_feature)

            segment_feature = visual_segment_features[seg_idx]

            moving_avg_count = 1

    story_segments[-1] = (story_segments[-1][0], len(visual_segment_features) - 1)
    story_segment_features.append(segment_feature)

    assert story_segment_count == len(story_segments)

    return story_segments, story_segment_features


def process_cluster(cluster: TopicCluster, args):
    ts15_stories = cluster.ts15s
    ts100_stories = cluster.ts100s

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

            segments, features = extract_segment_features(story)

            segment_features_per_story.append(torch.stack(features))
            segments_per_story.append(segments)
            all_segment_features.extend(features)

            bar()

    all_segment_features = torch.stack(all_segment_features)
    all_segment_features = F.normalize(all_segment_features, dim=1)

    # TS100

    ts100_segment_features = []

    with alive_bar(len(ts100_stories),
                   ctrl_c=False, title="Feature Extraction [ts100]",
                   length=25, force_tty=True, dual_line=True, receipt_text=True) as bar:
        for story in ts100_stories:
            bar.text = f'Story: {story.pk} -> {story.video}'

            _, features = extract_segment_features(story)

            ts100_segment_features.extend(features)
            bar()

    ts100_segment_features = torch.stack(ts100_segment_features)
    ts100_segment_features = F.normalize(ts100_segment_features, dim=1)

    for idx, (segments, segment_features) in enumerate(zip(segments_per_story, segment_features_per_story)):

        story = ts15_stories[idx]

        print(f"[{idx + 1}/{len(cluster.ts15s)}] Story: {story.headline} ({story.pk})")

        if len(segment_features) == 0:
            print('No features available ...')
            continue

        ts15_similarity_matrix = torch.matmul(segment_features, all_segment_features.t())
        ts15_similarity_sorted = ts15_similarity_matrix.sort(descending=True).values[:, :len(ts15_stories)]
        ts15_similarity_mean = ts15_similarity_sorted.mean(axis=1)
        ts15_similarity_mean_inv = F.normalize(1 - ts15_similarity_mean, dim=0)

        ts100_similarity_matrix = torch.matmul(segment_features, ts100_segment_features.t())
        ts100_similarity_sorted = ts100_similarity_matrix.sort(descending=True).values[:, :10]
        ts100_similarity_mean = ts100_similarity_sorted.mean(axis=1)
        ts100_similarity_mean = F.normalize(ts100_similarity_mean, dim=0)

        text_similarity_matrix = get_text_similarity_matrix(story, segment_features)

        if text_similarity_matrix is not None:
            text_similarity_mean = text_similarity_matrix.mean(axis=1)
            text_similarity_mean = F.normalize(text_similarity_mean, dim=0)

            segment_scores = (ts15_similarity_mean_inv + text_similarity_mean) / 3
        else:
            segment_scores = (ts15_similarity_mean_inv + ts100_similarity_mean) / 2

        segment_scores = F.normalize(segment_scores, dim=0)

        n_video_segments = segments[-1][1] + 1
        machine_summary = np.zeros(n_video_segments)
        machine_summary_scores = np.zeros(n_video_segments)

        threshold = ts15_similarity_mean_inv.mean()

        fig_1 = plt.figure(1, figsize=(18, 4))
        plt.subplots_adjust(bottom=0.18, left=0.05, right=0.98)

        segment_sp = fig_1.add_subplot(111)

        plt.xlabel('Segment')
        plt.ylabel('Score')

        x = range(n_video_segments)
        y_ts15_inv = flatten(
            [[seg_score] * (seg[1] - seg[0] + 1) for seg_score, seg in zip(ts15_similarity_mean_inv, segments)])
        y_ts15 = flatten(
            [[seg_score] * (seg[1] - seg[0] + 1) for seg_score, seg in zip(ts15_similarity_mean, segments)])
        y_ts100 = flatten(
            [[seg_score] * (seg[1] - seg[0] + 1) for seg_score, seg in zip(ts100_similarity_mean, segments)])
        y_text = flatten(
            [[seg_score] * (seg[1] - seg[0] + 1) for seg_score, seg in zip(text_similarity_mean, segments)])

        y_final = flatten([[seg_score] * (seg[1] - seg[0] + 1) for seg_score, seg in zip(segment_scores, segments)])

        # y_max = max(max(ts15_similarity_mean_inv), max(ts15_similarity_mean), max(text_similarity_mean))
        # y_min = min(min(ts15_similarity_mean_inv), min(ts15_similarity_mean), min(text_similarity_mean))

        y_max = max(max(ts15_similarity_mean_inv), max(ts100_similarity_mean), max(text_similarity_mean))
        y_min = min(min(ts15_similarity_mean_inv), min(ts100_similarity_mean), min(text_similarity_mean))

        y_axis_min = y_min - (y_max - y_min) * 0.1
        y_axis_max = y_max + (y_max - y_min) * 0.1
        plt.axis([0, n_video_segments - 1, y_axis_min, y_axis_max])

        plt.hlines(y=threshold, xmin=0, xmax=n_video_segments, linewidth=0.5, color=(0, 0, 0, 1))
        plt.vlines(flatten(segments), y_axis_min, y_axis_max, linestyles='dotted', colors='grey')

        segment_sp.plot(x, y_final, color='b', linewidth=0.8)
        segment_sp.plot(x, y_ts15_inv, color='r', linewidth=0.5)
        # segment_sp.plot(x, y_ts15, color='r', linestyle='dashed', linewidth=0.5)
        segment_sp.plot(x, y_ts100, color='g', linewidth=0.5)
        segment_sp.plot(x, y_text, color='c', linewidth=0.5)
        plt.xticks(range(0, n_video_segments, 5))

        plt.fill_between(x, y_axis_min, y_axis_max, where=(y_final > threshold.numpy()), color='b', alpha=.1)

        summary_video = []
        for itr, score in enumerate(segment_scores):
            start, end = segments[itr]

            if end >= start:
                machine_summary_scores[start: end + 1] = score
                if score >= threshold:
                    machine_summary[start: end + 1] = 1
                    if idx % 5 == 0 and args.pseudo_video_dir:
                        frames = story.frames[segment_idx_to_frame_idx(0, start): segment_idx_to_frame_idx(0, end)]
                        summary_video.append(torch.tensor(np.array(read_images(frames))))

        plt.show()

        if idx % 5 == 0 and args.pseudo_video_dir and len(summary_video) > 1:
            summary_video = torch.cat(summary_video, dim=0)

            pseudo_video_dir = Path(args.pseudo_video_dir, str(cluster.index))

            create_dir(pseudo_video_dir)

            io.write_video(
                os.path.join(str(pseudo_video_dir), "{}.mp4".format(story.pk)),
                summary_video,
                25,
            )

        redis_summary = db.List('pseudo:summary:' + story.pk)
        redis_scores = db.List('pseudo:scores:' + story.pk)

        redis_summary.extend(machine_summary.tolist())
        redis_scores.extend(machine_summary_scores.tolist())


def main(args):
    if args.index:
        clusters = [TopicCluster.find(TopicCluster.index == index).first() for index in args.index]
        assert all(cluster.features == 1 for cluster in clusters)

    else:
        clusters = TopicCluster.find(TopicCluster.features == 1).sort_by('-index').all()
        random.shuffle(clusters)

    for cluster in clusters:
        print(f'----- Cluster {cluster.index} -----')
        process_cluster(cluster, args)
        print()

    sys.exit()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
