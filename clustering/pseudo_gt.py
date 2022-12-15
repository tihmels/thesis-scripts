import matplotlib
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.io as io

matplotlib.use('TkAgg')

from common.utils import read_images, frame_idx_to_time
from database import rai, db
from database.config import RAI_SEG_PREFIX, RAI_TEXT_PREFIX
from database.model import TopicCluster


def segment_idx_to_frame_idx(story_start_idx, segment_idx):
    return story_start_idx + (segment_idx * 3 * 16)


def segment_to_frame_range(story_start_idx, first_segment_idx: int, last_segment_idx: int = None):
    last_segment_idx = last_segment_idx if last_segment_idx else first_segment_idx + 1
    return (segment_idx_to_frame_idx(story_start_idx, first_segment_idx),
            segment_idx_to_frame_idx(story_start_idx, last_segment_idx))


def process_cluster(cluster: TopicCluster):
    ts15_stories = cluster.ts15s
    ts100_stories = cluster.ts100s

    ts15_summaries = {}

    segment_features_per_story = []
    segments_per_story = []
    all_segment_features = []

    for story in ts15_stories:
        story_segment_features = []
        story_segment_count = 1
        story_segments = [(0, 0)]

        segment_features = torch.tensor(rai.get_tensor(RAI_SEG_PREFIX + story.pk))
        segment_features = F.normalize(segment_features, dim=1)

        similarity_matrix = torch.matmul(segment_features, segment_features.t())
        similarity_means = similarity_matrix.mean(axis=1)
        max_similarity = similarity_means.max()

        segment_feature = segment_features[0]
        reference_feature = segment_features[0]
        moving_avg_count = 1

        for seg_idx in range(1, len(segment_features)):
            similarity = torch.matmul(segment_features[seg_idx], reference_feature.t())

            start, end = segment_to_frame_range(story.first_frame_idx, seg_idx)
            start_time = frame_idx_to_time(start)
            end_time = frame_idx_to_time(end)

            if similarity > 0.85 * max_similarity:
                segment_feature += segment_features[seg_idx]
                moving_avg_count += 1
                reference_feature = segment_feature / moving_avg_count
            else:
                story_segment_count += 1

                story_segments[-1] = (story_segments[-1][0], seg_idx - 1)
                story_segments.append((seg_idx, seg_idx))

                story_segment_features.append(segment_feature / moving_avg_count)
                all_segment_features.append(segment_feature / moving_avg_count)

                segment_feature = segment_features[seg_idx]
                reference_feature = segment_features[seg_idx]

                moving_avg_count = 1

        if moving_avg_count > 1:
            story_segments[-1] = (story_segments[-1][0], len(segment_features))

            story_segment_features.append(segment_feature / moving_avg_count)
            all_segment_features.append(segment_feature / moving_avg_count)

        assert story_segment_count == len(story_segments)
        segment_features_per_story.append(torch.stack(story_segment_features))
        segments_per_story.append(story_segments)

    all_segment_features = torch.stack(all_segment_features)
    all_segment_features = F.normalize(all_segment_features, dim=1)

    # TS100

    ts100_segment_features = []

    for story in ts100_stories:
        segment_features = torch.tensor(rai.get_tensor(RAI_SEG_PREFIX + story.pk))
        segment_features = F.normalize(segment_features, dim=1)

        similarity_matrix = torch.matmul(segment_features, segment_features.t())
        similarity_means = similarity_matrix.mean(axis=1)
        max_similarity = similarity_means.max()

        segment_feature = segment_features[0]
        reference_feature = segment_features[0]
        moving_avg_count = 1

        for seg_idx in range(1, len(segment_features)):
            similarity = torch.matmul(segment_features[seg_idx], reference_feature.t())

            if similarity > 0.85 * max_similarity:
                segment_feature += segment_features[seg_idx]
                moving_avg_count += 1
                reference_feature = segment_feature / moving_avg_count
            else:
                ts100_segment_features.append(segment_feature / moving_avg_count)

                segment_feature = segment_features[seg_idx]
                reference_feature = segment_features[seg_idx]

                moving_avg_count = 1

        if moving_avg_count > 1:
            ts100_segment_features.append(segment_feature / moving_avg_count)

    ts100_segment_features = torch.stack(ts100_segment_features)
    ts100_segment_features = F.normalize(ts100_segment_features, dim=1)

    for idx, (segments, segment_features) in enumerate(zip(segments_per_story, segment_features_per_story)):

        story = ts15_stories[idx]

        print(f"[{story.pk}] Story: ", story.headline)
        print(f"[{story.pk}] Number of segments: ", len(segments))

        if len(segment_features) == 0:
            print('No features available ...')
            continue

        segment_features = F.normalize(segment_features, dim=1)
        text_similarity_mean = None

        if rai.tensor_exists(RAI_TEXT_PREFIX + story.pk):
            text_features = torch.tensor(rai.get_tensor(RAI_TEXT_PREFIX + story.pk))
            text_features = F.normalize(text_features, dim=1)

            text_similarity_matrix = torch.matmul(segment_features, text_features.t())
            text_similarity_mean = text_similarity_matrix.mean(axis=1)

        v_similarity_matrix = torch.matmul(segment_features, all_segment_features.t())
        v_similarity_mean = v_similarity_matrix.mean(axis=1)

        ts100_similarity_matrix = torch.matmul(segment_features, ts100_segment_features.t())
        ts100_similarity_mean = ts100_similarity_matrix.mean(axis=1)

        # Combine both the similarity matrices
        if text_similarity_mean is not None:
            segment_scores = (v_similarity_mean + ts100_similarity_mean + text_similarity_mean) / 3
        else:
            segment_scores = v_similarity_mean

        segment_scores = F.normalize(segment_scores, dim=0)

        n_video_segments = segments[-1][1] + 1
        machine_summary = np.zeros(n_video_segments)
        machine_summary_scores = np.zeros(n_video_segments)

        threshold = 0.8 * segment_scores.max()

        summary_video = []
        for itr, score in enumerate(segment_scores):
            start, end = segments[itr]

            start_, end_ = segment_to_frame_range(story.first_frame_idx, start, end)
            start_time = frame_idx_to_time(start_)
            end_time = frame_idx_to_time(end_)

            if end >= start:
                machine_summary_scores[start: end + 1] = score
                if score >= threshold:
                    machine_summary[start: end + 1] = 1
                    if idx % 10 == 0 and story is not None:
                        frames = story.frames[start * 16 * 3: end * 16 * 3]
                        summary_video.append(torch.tensor(np.array(read_images(frames))))

        if idx % 10 == 0 and story is not None:
            summary_video = torch.cat(summary_video, dim=0)

            io.write_video(
                os.path.join("/Users/tihmels/Desktop/summaries/", "{}_summary.mp4".format(story.headline)),
                summary_video,
                25,
            )

        redis_summary = db.List('pseudo:summary:' + story.pk)
        redis_scores = db.List('pseudo:scores:' + story.pk)

        redis_summary.extend(machine_summary.tolist())
        redis_scores.extend(machine_summary_scores.tolist())


def main():
    cluster = TopicCluster.find(TopicCluster.features == 1).sort_by('-index').first()
    process_cluster(cluster)


if __name__ == "__main__":
    main()
