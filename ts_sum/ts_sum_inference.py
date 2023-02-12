import math
import os
from argparse import ArgumentParser
from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from shutil import copy

import matplotlib
import torch
import torchvision.io as io
from alive_progress import alive_bar
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from torchvision.transforms import transforms
from tqdm import tqdm

from common.VAO import VAO
from common.utils import read_images, Range, flatten, create_dir
from database import db, rai
from database.model import MainVideo, get_sum_key, get_score_key
from ts_sum.evaluate_and_log import evaluate_summary
from ts_sum.knapsack import knapsack_ortools
from ts_sum.ts_sum_utils import get_last_checkpoint, Logger, AverageMeter
from ts_sum.vsum import VSum

matplotlib.use('TkAgg')

parser = ArgumentParser()

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="/Users/tihmels/Scripts/thesis-scripts/ts_sum/vsum_checkpoint/exp_model_1_bs_8_lr_0.001_nframes_896_nfps_32_nheads_8_nenc_6_dropout_0.1_finetune_False",
    help="checkpoint model folder",
)
parser.add_argument(
    "--log_root", type=str, default="vsum_tboard_log", help="log dir root"
)
parser.add_argument(
    "--log_name", default="exp", help="name of the experiment for checkpoints and logs",
)
parser.add_argument(
    "--log_videos",
    dest="log_videos",
    action="store_true",
    help="Log top 10 and bottom 10 result videos",
)
parser.add_argument(
    "-out_dir",
    "--out_dir",
    default="./out_sum",
    type=str,
    help="folder for result videos",
)
parser.add_argument("--word2vec_path", type=str, default="data/word2vec.pth", help="")
parser.add_argument(
    "--pretrain_cnn_path",
    type=str,
    default="./pretrained_weights/s3d_howto100m.pth",
    help="",
)
parser.add_argument(
    "--window_len", type=int, default=16, help="window len"
)
parser.add_argument(
    "--proportion", type=float, default=0.12, help="percentage value to reduce to main edition to"
)
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help='Tagesschau video file(s)')


def visualize_picks(shots, frame_scores, picks):
    plt.figure(figsize=(25, 12))

    plt.title("Frame Scores", fontsize=10)

    plt.xlabel("Frame")
    plt.ylabel("Score")

    y_min = 0
    y_max = max(frame_scores)
    y_max = y_max + y_max * 0.1

    x_range = list(range(len(frame_scores)))

    plt.xlim([0, len(frame_scores)])
    plt.plot(x_range, frame_scores)

    # plt.vlines([shot.last_frame_idx for shot in shots[:-1]], 0, y_max, colors='grey')

    for pick in picks:
        shot = shots[pick]
        shot_range = range(shot.first_frame_idx, shot.last_frame_idx)

        plt.fill_between(shot_range, y_min, y_max, color='b', alpha=.1)

    plt.xticks(range(0, len(frame_scores), 1000))
    plt.show()


def log_summaries(
        tb_logger,
        out_dir,
        video_summaries,
        log_videos=False,
):
    # Compute scores
    f_scores = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    # Write to table
    table = PrettyTable()
    table.title = "Video Summary Eval Results"
    table.field_names = ["ID", "n-stories", "n-stories-eval", "total-eval", "F-score", "Precision", "Recall"]
    table.float_format = "1.3"

    for video_pk in video_summaries.keys():
        video = MainVideo.find(MainVideo.pk == video_pk).first()

        eval_stories = [story for story in video.stories if rai.tensor_exists(get_sum_key(story.pk))]

        pseudo_summaries = [db.List(get_sum_key(story.pk)).as_list() for story in eval_stories]
        pseudo_scores = [db.List(get_score_key(story.pk)).as_list() for story in eval_stories]

        video_summary = video_summaries[video_pk]

        scores = [db.List(get_score_key(story.pk)).as_list() for story in video.stories]
        scores = [list(map(float, score)) for score in scores]

        # evaluate_summary() to be done
        f_score, prec, recall = evaluate_summary(video_summaries['machine_summary'], pseudo_summaries)

        f_scores.update(f_score)
        precisions.update(prec)
        recalls.update(recall)

        n_eval_frames = sum([len(story.frames) for story in eval_stories])
        eval_proportion = n_eval_frames / len(video.frames)

        print(f"F-Score: {f_score}")

        table.add_row([video_pk, len(video.stories), len(eval_stories), eval_proportion, f_score, prec, recall])

    logs = OrderedDict()
    logs["F-Score"] = f_scores.avg
    logs["Precision"] = precisions.avg
    logs["Recall"] = recalls.avg

    # Write logger
    for key, value in logs.items():
        tb_logger.log_scalar(value, key, "Summary Generation")
    tb_logger.flush()

    # Write table
    table.add_row(["mean", f_scores.avg, precisions.avg, recalls.avg])
    tqdm.write(str(table))

    if log_videos:
        for video_pk in video_summaries.keys():
            video = MainVideo.find(MainVideo.pk == video_pk).first()

            shot_picks = video_summaries[video_pk]['shot_picks']

            summary_shots = [shot for idx, shot in enumerate(video.shots) if idx in shot_picks]

            summary_frames = flatten([shot.frames for shot in summary_shots])

            summary_video = torch.tensor(read_images(summary_frames))

            parent_out = Path(out_dir, video_pk)
            create_dir(parent_out)

            io.write_video(
                str(Path(parent_out, f'{video.pk}-SUM.mp4')),
                summary_video,
                25,
            )

            summary_keyframes = [shot.keyframe for shot in summary_shots]
            non_summary_keyframes = [shot.keyframe for idx, shot in enumerate(video.shots) if idx not in shot_picks]

            kf_path = Path(parent_out, 'sum_keyframes')
            create_dir(kf_path)

            for keyframe in summary_keyframes:
                copy(keyframe, kf_path)

            non_kf_path = Path(parent_out, 'non_sum_keyframes')
            create_dir(non_kf_path)

            for keyframe in non_summary_keyframes:
                copy(keyframe, non_kf_path)

    # CHECK Whats happening here
    # Sort scores and visualize
    # sorted_ids = sorted(
    #     video_summaries, key=lambda x: (video_summaries[x]["segment_scores"])
    # )
    #
    # # Top 10 and bottom 10 scoring videos
    # top_bottom_10 = sorted_ids[-10:]
    # top_bottom_10.extend(sorted_ids[:10])


def rename_dict(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def seg_idx_to_frame_range(seg_idx, window_len=16):
    return Range(seg_idx * window_len, (seg_idx + 1) * window_len - 1)


def range_overlap(r1: Range, r2: Range):
    latest_start = max(r1.start, r2.start)
    earliest_end = min(r1.end, r2.end)
    delta = earliest_end - latest_start
    overlap = max(0, delta)

    return overlap


def main(args):
    videos = [MainVideo.find(MainVideo.pk == VAO(file).id).first() for file in args.files]

    if args.checkpoint_dir[-3:] == "tar":
        args.log_name = args.checkpoint_dir.split("/")[-2] + "_eval"
        checkpoint_path = args.checkpoint_dir
    else:
        args.log_name = args.checkpoint_dir.split("/")[-1] + "_eval"
        checkpoint_path = get_last_checkpoint(args.checkpoint_dir)

    out_dir = os.path.join(args.out_dir, args.log_name)
    os.makedirs(out_dir, exist_ok=True)

    tb_logdir = os.path.join(args.log_root, args.log_name)
    os.makedirs(tb_logdir, exist_ok=True)
    tb_logger = Logger(tb_logdir)

    video_summaries = {}

    model = VSum(space_to_depth=True, window_len=args.window_len, word2vec_path=args.word2vec_path)
    model = model.eval()

    if checkpoint_path:
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)

        # state_dict = rename_dict(checkpoint["state_dict"])

        model.load_state_dict(checkpoint["state_dict"])

        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_path, checkpoint["epoch"]
            )
        )

    with torch.no_grad():
        for itr, video in enumerate(videos):
            print("Generating summary for: ", video.pk)

            frames = read_images(video.frames[::5])

            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((224, 224))]
            )

            frames = torch.stack([transform(frame) for frame in frames])

            window_len = args.window_len

            extra_frames = window_len - (len(frames) % window_len)
            frames = torch.cat((frames, frames[-extra_frames:]), dim=0)

            n_segments = int(frames.shape[0] / window_len)

            # [B, C, H, W] -> [B, T, C, H, W]
            frames = frames.view(n_segments, window_len, 3, 224, 224)

            # [B, T, C, H, W] -> [B, C, T, H, W]
            frames = frames.permute(0, 2, 1, 3, 4)

            segment_scores = []

            with alive_bar(n_segments, ctrl_c=False, title=f'{video.pk}', length=35) as bar:
                for segment in frames:
                    # batch = segment.unsqueeze(0).cuda() # <---
                    batch = segment.unsqueeze(0)
                    # _, score = model(batch) # <---
                    score = torch.rand((1, 1))
                    segment_scores.append(score.view(-1))
                    bar()

            segment_scores = torch.stack(segment_scores)

            shots = video.shots

            shot_scores = calc_shot_scores(segment_scores, shots, window_len)

            shot_n_frames = [(shot.last_frame_idx - shot.first_frame_idx) + 1 for shot in shots]
            capacity = int(math.floor(len(video.frames) * args.proportion))

            shot_picks = knapsack_ortools(shot_scores, shot_n_frames, capacity)

            shot_score_frames = flatten([repeat(shot_scores[idx], shot_n_frames[idx]) for idx in range(len(shots))])

            visualize_picks(shots, shot_score_frames, shot_picks)

            binary_frame_summary = flatten([
                list(repeat(1 if idx in shot_picks else 0, shot_n_frames[idx])) for idx, shot in enumerate(shots)])

            video_summaries[video.pk] = {}
            video_summaries[video.pk]["shot_scores"] = shot_scores
            video_summaries[video.pk]["shot_picks"] = shot_picks
            video_summaries[video.pk]["segment_scores"] = segment_scores
            video_summaries[video.pk]["machine_summary"] = binary_frame_summary

        # Calculate scores and log videos
        log_summaries(
            tb_logger,
            "/Users/tihmels/Desktop/SummaryLogs",
            video_summaries,
            log_videos=args.log_videos,
        )


def calc_shot_scores(segment_scores, shots, window_len):
    shot_scores = []

    for shot in shots:
        shot_range = Range(shot.first_frame_idx, shot.last_frame_idx)

        segment_idxs = [idx for idx in range(len(segment_scores)) if
                        range_overlap(shot_range, seg_idx_to_frame_range(idx)) > 0]

        total_score = 0

        for seg_idx in segment_idxs:
            score = segment_scores[seg_idx].cpu().detach().numpy()
            score_per_frame = score / window_len

            n_frames_overlap = range_overlap(shot_range, seg_idx_to_frame_range(seg_idx))
            total_score += score_per_frame * n_frames_overlap

        # shot_scores.append(total_score)
        shot_scores.append(total_score / (shot.last_frame_idx - shot.first_frame_idx))

    return shot_scores


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
