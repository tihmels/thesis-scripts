import math
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchvision.io as io

from common.utils import read_images, frame_idx_to_time
from database import rai
from database.config import RAI_SEG_PREFIX, RAI_TEXT_PREFIX
from database.model import TopicCluster


def convert_segment_to_frame_idx(story_start_idx, segment_idx):
    return story_start_idx + (segment_idx * 3 * 16)


def convert_range_to_frame_idx(story_start_idx, segment):
    start, end = segment
    return (convert_segment_to_frame_idx(story_start_idx, start), convert_segment_to_frame_idx(story_start_idx, end))


def process_cluster(cluster: TopicCluster):
    ts15_stories = cluster.ts15s
    all_summaries = {}

    ts15_features = []  # segment features per video
    all_features = []  # all segment features
    video_segments = []  # segments for all videos

    for story in ts15_stories:
        segment_features = []
        segment_count = 1
        segments = [(0, 0)]

        total_frames = story.last_frame_idx - story.first_frame_idx
        total_frames_subsampled = int(total_frames / 3)
        n_segments = math.ceil(total_frames_subsampled / 16)

        features = torch.tensor(rai.get_tensor(RAI_SEG_PREFIX + story.pk))
        features = F.normalize(features, dim=1)

        # Find max vid feature similarity
        vid_feat_mat = torch.matmul(features, features.t())
        vid_feat_sim_mean = vid_feat_mat.mean(dim=1)
        max_sim = vid_feat_sim_mean.max()

        avg_feature = features[0]
        start_feature = features[0]
        moving_avg_count = 1

        for i in range(1, len(features)):
            sim = torch.matmul(features[i], start_feature.t())
            start, end = convert_range_to_frame_idx(story.first_frame_idx, (i, i + 1))
            start_time, end_time = frame_idx_to_time(start), frame_idx_to_time(end)
            if sim > max_sim:
                avg_feature += features[i]
                start_feature = features[i]
                moving_avg_count += 1
            else:
                segment_count += 1
                segments[len(segments) - 1] = (
                    segments[len(segments) - 1][0],
                    i - 1,
                )
                segments.append((i, i))
                segment_features.append(avg_feature / moving_avg_count)
                all_features.append(avg_feature)
                avg_feature = features[i]
                start_feature = features[i]
                moving_avg_count = 1

        if moving_avg_count > 1:
            segments[len(segments) - 1] = (
                segments[len(segments) - 1][0],
                i,
            )
            segment_features.append(avg_feature / moving_avg_count)
            all_features.append(avg_feature)

        assert segment_count == len(segments)
        ts15_features.append(torch.stack(segment_features))
        video_segments.append(segments)

        frame_segments = [convert_segment_to_frame_idx(seg) for seg in segments]

        print()

    all_features = torch.stack(all_features)
    all_features = F.normalize(all_features, dim=1)

    for idx, (segments, segment_features) in enumerate(zip(video_segments, ts15_features)):

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
            torch.matmul(segment_features, all_features.t()).detach().cpu()
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

        all_summaries[story.pk] = {}
        all_summaries[story.pk][
            "machine_summary"
        ] = machine_summary.tolist()
        all_summaries[story.pk][
            "machine_summary_scores"
        ] = machine_summary_scores.tolist()


def main():
    cluster = TopicCluster.find(TopicCluster.features == 1).first()
    process_cluster(cluster)


if __name__ == "__main__":
    main()
