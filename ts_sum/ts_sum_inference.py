import random

import numpy as np
import os
import torch
import torch.nn as nn
from argparse import ArgumentParser
from prettytable import PrettyTable
from torchvision.transforms import transforms

from common.utils import read_images, flatten
from database import rai, db
from database.config import get_sum_key, get_score_key
from database.model import MainVideo
from ts_sum import s3dg
from ts_sum.ts_sum_utils import get_last_checkpoint, Logger, AverageMeter

parser = ArgumentParser(description="PyTorch ASR Video Segment MIL-NCE")

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


def log_scores(
        tb_logger,
        annt_dir,
        video_dir,
        video_frames_dir,
        out_dir,
        all_video_summaries,
        epoch=0,
        log_videos=False,
):
    # Compute scores
    f_scores = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    # Write to table
    table = PrettyTable()
    table.title = "Eval result of epoch {}".format(epoch)
    table.field_names = ["ID", "F-score", "Precision", "Recall"]
    table.float_format = "1.3"

    remove_keys = []
    for video_pk in all_video_summaries.keys():
        video = MainVideo.find(MainVideo.pk == video_pk).first()

        gt_summary = [db.List(get_sum_key(story.pk)).as_list() for story in video.stories]
        gt_summary = [list(map(int, map(float, label))) for label in gt_summary]

        scores = [db.List(get_score_key(story.pk)).as_list() for story in video.stories]
        scores = [list(map(float, score)) for score in scores]

        print(
            "GT summary shape, machine summary shape: ",
            len(gt_summary),
            len(all_video_summaries[video_pk]["machine_summary"]),
        )

        all_video_summaries[video_pk]["machine_summary"] = all_video_summaries[
                                                               video_pk
                                                           ]["machine_summary"][:len(gt_summary)]


def rename_dict(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def main(args):
    videos = MainVideo.find().all()
    videos = [video for video in videos if
              all([rai.tensor_exists(get_sum_key(story.pk)) for story in video.stories])]

    if args.checkpoint_dir[-3:] == "tar":
        args.log_name = args.checkpoint_dir.split("/")[-2] + "_eval"
        checkpoint_path = args.checkpoint_dir
    else:
        args.log_name = args.checkpoint_dir.split("/")[-1] + "_eval"
        checkpoint_path = get_last_checkpoint(args.checkpoint_dir)

    args.out_dir = os.path.join(args.out_dir, args.log_name)
    os.makedirs(args.out_dir, exist_ok=True)

    # start a logger
    tb_logdir = os.path.join(args.log_root, args.log_name)
    os.makedirs(tb_logdir, exist_ok=True)
    tb_logger = Logger(tb_logdir)

    all_video_summaries = {}

    model = s3dg.VSum(space_to_depth=True, word2vec_path=args.word2vec_path)
    model = model.eval()
    print("Created model")

    if checkpoint_path:
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)

        #state_dict = rename_dict(checkpoint["state_dict"])

        model.load_state_dict(checkpoint["state_dict"])

        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_path, checkpoint["epoch"]
            )
        )

        with torch.no_grad():
            for itr, video in enumerate(random.sample(videos, 2)):
                print("Starting video: ", video.pk)

                frames = flatten([story.frames for story in video.stories])
                frames = read_images(frames[::2])

                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Resize((224, 224))]
                )

                frames = torch.stack([transform(frame) for frame in frames])

                window_len = 32

                # Pad video with extra frames to ensure its divisible by window_len
                extra_frames = window_len - (len(frames) % window_len)
                frames = torch.cat((frames, frames[-extra_frames:]), dim=0)

                n_segments = int(frames.shape[0] / window_len)

                print("Number of video segments: ", n_segments)

                # [B, C, H, W] -> [B, T, C, H, W]
                frames = frames.view(n_segments, window_len, 3, 224, 224)

                # [B, T, C, H, W] -> [B, C, T, H, W]
                frames = frames.permute(0, 2, 1, 3, 4)

                scores = []

                for segment in frames:
                    batch = segment.unsqueeze(0)
                    _, score = model(batch)
                    scores.append(score.view(-1))

                scores = torch.stack(scores)

                summary_frames = nn.functional.softmax(scores, dim=1)[:, 1]
                summary_frames[summary_frames > 0.5] = 1
                summary_frames = np.repeat(summary_frames.detach().cpu().numpy(), 32)
                print("Shape of summary frames:", summary_frames.shape)

                all_video_summaries[video.pk] = {}
                all_video_summaries[video.pk]["machine_summary"] = summary_frames.tolist()

            # Calculate scores and log videos
            log_scores(
                tb_logger,
                args.annt_dir,
                "/home/medhini/video_summarization/task_video_sum/datasets/how_to_summary_videos",
                args.video_frames_dir,
                args.out_dir,
                all_video_summaries,
                log_videos=args.log_videos,
            )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
