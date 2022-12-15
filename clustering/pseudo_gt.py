import math
import matplotlib
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.io as io

matplotlib.use('TkAgg')

from common.utils import read_images, frame_idx_to_time
from database import rai
from database.config import RAI_SEG_PREFIX, RAI_TEXT_PREFIX
from database.model import TopicCluster


def segment_idx_to_frame_idx(story_start_idx, segment_idx):
    return story_start_idx + (segment_idx * 3 * 16)


def segment_to_frame_range(story_start_idx, segment):
    start, end = segment
    return (segment_idx_to_frame_idx(story_start_idx, start),
            segment_idx_to_frame_idx(story_start_idx, end))


def process_cluster(cluster: TopicCluster):
    ts15_stories = cluster.ts15s
    ts15_summaries = {}

    segment_features_per_story = []
    all_segment_features = []
    all_segments = []

    for story in ts15_stories[2:]:
        story_segment_features = []
        story_segment_count = 1
        story_segments = [(0, 0)]

        features = torch.tensor(rai.get_tensor(RAI_SEG_PREFIX + story.pk))
        features = F.normalize(features, dim=1)

        # Find max vid feature similarity
        similarity_matrix = torch.matmul(features, features.t())
        similarity_means = similarity_matrix.mean(axis=1)
        max_similarity = similarity_means.max()

        segment_feature = features[0]
        reference_feature = features[0]
        moving_avg_count = 1

        for i in range(1, len(features)):
            similarity = torch.matmul(features[i], reference_feature.t())

            start, end = segment_to_frame_range(story.first_frame_idx, (i, i + 1))
            start_time = frame_idx_to_time(start)
            end_time = frame_idx_to_time(end)

            if similarity > max_similarity:
                segment_feature += features[i]
                moving_avg_count += 1
                reference_feature = segment_feature / moving_avg_count
            else:
                story_segment_count += 1

                story_segments[-1] = (story_segments[-1][0], i - 1)
                story_segments.append((i, i))

                story_segment_features.append(segment_feature / moving_avg_count)
                all_segment_features.append(segment_feature / moving_avg_count)

                segment_feature = features[i]
                reference_feature = features[i]
                moving_avg_count = 1

        if moving_avg_count > 1:
            story_segments[-1] = (story_segments[-1][0], len(features))
            story_segment_features.append(segment_feature / moving_avg_count)
            all_segment_features.append(segment_feature / moving_avg_count)

        assert story_segment_count == len(story_segments)
        segment_features_per_story.append(torch.stack(story_segment_features))
        all_segments.append(story_segments)

        frame_segments = [segment_to_frame_range(story.first_frame_idx, seg) for seg in story_segments]

        print()

    all_segment_features = torch.stack(all_segment_features)
    # all_segment_features = F.normalize(all_segment_features, dim=1)

    for idx, (segments, segment_features) in enumerate(zip(video_segments, segment_features_per_story)):

        story = ts15_stories[idx]

        print(f"[{story.pk}] Story: ", story.headline)
        print(f"[{story.pk}] Number of segments: ", len(segments))

        if len(segment_features) == 0:
            print('No features available ...')
            continue

        segment_features = F.normalize(segment_features, dim=1)

        asr_similarity_matrix = None
        if rai.tensor_exists(RAI_TEXT_PREFIX + story.pk):
            asr_features = torch.tensor(rai.get_tensor(RAI_TEXT_PREFIX + story.pk))
            print("ASR shape: ", asr_features.shape)

            asr_features = F.normalize(asr_features, dim=1)
            asr_similarity_matrix = (
                torch.matmul(segment_features, asr_features.t()).detach().cpu()
            )
            asr_similarity_matrix = asr_similarity_matrix.mean(axis=1)

        v_similarity_matrix = (
            torch.matmul(segment_features, all_segment_features.t()).detach().cpu()
        )
        v_similarity_matrix = v_similarity_matrix.mean(axis=1)

        # Combine both the similarity matrices
        if asr_similarity_matrix is not None:
            segment_scores = (v_similarity_matrix + asr_similarity_matrix) / 2
        else:
            segment_scores = v_similarity_matrix

        segment_scores = F.normalize(segment_scores, dim=0)

        n_video_segments = segments[-1][1] + 1
        machine_summary = np.zeros(n_video_segments)
        machine_summary_scores = np.zeros(n_video_segments)
        print("Machine summary shape: ", machine_summary.shape)

        threshold = 0.85 * segment_scores.max()
        print("Threshold: ", threshold)
        summary_video = []
        for itr, score in enumerate(segment_scores):
            segment = segments[itr]
            if segment[1] >= segment[0]:
                machine_summary_scores[segment[0]: (segment[1] + 1)] = score
                if score >= threshold:
                    machine_summary[segment[0]: (segment[1] + 1)] = 1
                    if idx == 0 and story is not None:
                        summary_video.append(
                            torch.tensor(np.array(read_images(story.frames[segment[0] * 16: (segment[1] + 1) * 16])))
                        )

        if idx == 0 and story is not None:
            summary_video = torch.cat(summary_video, dim=0)
            print("Summary video shape: ", summary_video.shape)

            io.write_video(
                os.path.join(
                    "/Users/tihmels/Desktop/summaries/",
                    "{}_summary.mp4".format(story.headline),
                ),
                summary_video,
                25,
            )

        ts15_summaries[story.pk] = {}
        ts15_summaries[story.pk][
            "machine_summary"
        ] = machine_summary.tolist()
        ts15_summaries[story.pk][
            "machine_summary_scores"
        ] = machine_summary_scores.tolist()


def main():
    cluster = TopicCluster.find(TopicCluster.features == 1).first()
    process_cluster(cluster)


if __name__ == "__main__":
    main()
