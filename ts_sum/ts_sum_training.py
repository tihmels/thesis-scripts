from logging import Logger

import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from argparse import ArgumentParser

import s3dg
from ts_sum.video_loader import VSum_DataLoader

parser = ArgumentParser('Setup RedisAI DB')
parser.add_argument("--seed", default=1, type=int, help="seed for initializing training.")
parser.add_argument("--model_type", "-m", default=1, type=int, help="(1) VSum_Trans (2) VSum_MLP")
parser.add_argument("--optimizer", type=str, default="adam", choices=['adam', 'sgd'], help="opt algorithm")
parser.add_argument("--num_class", type=int, default=512, help="upper epoch limit")
parser.add_argument("--word2vec_path", type=str, default="", help="")
parser.add_argument("--weight_init", type=str, default="uniform", help="CNN weights inits")
parser.add_argument("--dropout", "--dropout", default=0.1, type=float, help="Dropout")
parser.add_argument("--min_time", type=float, default=5.0, help="")
parser.add_argument("--fps", type=int, default=12, help="")
parser.add_argument("--heads", "-heads", default=8, type=int, help="number of transformer heads")
parser.add_argument("--finetune", dest="finetune", action="store_true", help="finetune S3D")
parser.add_argument("--video_size", type=int, default=224, help="image size")
parser.add_argument("--crop_only", type=int, default=1, help="random seed")
parser.add_argument("--centercrop", type=int, default=0, help="random seed")
parser.add_argument("--random_flip", type=int, default=1, help="random seed")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--num_thread_reader", type=int, default=20, help="")
parser.add_argument("--rank", default=0, type=int, help="Rank.")
parser.add_argument(
    "--batch_size_eval", type=int, default=16, help="batch size eval"
)
parser.add_argument(
    "--pin_memory", dest="pin_memory", action="store_true", help="use pin_memory"
)
parser.add_argument(
    "--num_candidates", type=int, default=1, help="num candidates for MILNCE loss"
)
parser.add_argument(
    "--enc_layers",
    "-enc_layers",
    default=24,
    type=int,
    help="number of layers in transformer encoder",
)
parser.add_argument(
    "--num_frames",
    type=int,
    default=896,
    help="number of frames in each video clip",
)
parser.add_argument(
    "--num_frames_per_segment",
    type=int,
    default=32,
    help="number of frames in each segment",
)
parser.add_argument(
    "--lrv",
    "--learning-rate-vsum",
    default=0.001,
    type=float,
    metavar="LRV",
    help="initial learning rate",
    dest="lrv",
)
parser.add_argument(
    "--weight_decay",
    "--wd",
    default=0.00001,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="vsum_checkpoint",
    help="checkpoint model folder",
)
parser.add_argument(
    "--log_root", type=str, default="vsum_tboard_log", help="log dir root"
)
parser.add_argument(
    "--log_name",
    default="exp",
    help="name of the experiment for checkpoints and logs",
)


def create_logger(args):
    args.log_name = "{}_model_{}_bs_{}_lr_{}_nframes_{}_nfps_{}_nheads_{}_nenc_{}_dropout_{}_finetune_{}".format(
        args.log_name,
        args.model_type,
        args.batch_size,
        args.lrv,
        args.num_frames,
        args.num_frames_per_segment,
        args.heads,
        args.enc_layers,
        args.dropout,
        args.finetune,
    )
    tb_logdir = os.path.join(args.log_root, args.log_name)
    tb_logger = Logger(tb_logdir)
    if args.rank == 0:
        os.makedirs(tb_logdir, exist_ok=True)

    return tb_logger


def log(output, args):
    print(output)
    with open(
            os.path.join(
                os.path.dirname(__file__), "vsum_ouptut_log", args.log_name + ".txt"
            ),
            "a",
    ) as f:
        f.write(output + "\n")


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = create_model(args)

    for name, param in model.named_parameters():
        if "base" in name:
            param.requires_grad = False
        if "mixed_5" in name and args.finetune:
            param.requires_grad = True

    model = torch.nn.DataParallel(model)

    train_dataset = VSum_DataLoader(
        min_time=args.min_time,
        fps=args.fps,
        num_frames=args.num_frames,
        num_frames_per_segment=args.num_frames_per_segment,
        size=args.video_size,
        crop_only=args.crop_only,
        center_crop=args.centercrop,
        random_left_right_flip=args.random_flip,
        num_candidates=args.num_candidates,
        video_only=True,
    )
    # Test data loading code
    test_dataset = VSum_DataLoader(
        min_time=args.min_time,
        fps=args.fps,
        num_frames=args.num_frames,
        num_frames_per_segment=args.num_frames_per_segment,
        size=args.video_size,
        crop_only=args.crop_only,
        center_crop=args.centercrop,
        random_left_right_flip=args.random_flip,
        num_candidates=args.num_candidates,
        video_only=True,
        dataset="wikihow",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_thread_reader,
        pin_memory=args.pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_eval,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_thread_reader
    )

    logger = create_logger(args)

    criterion_c = nn.MSELoss(reduction="none")
    criterion_r = nn.MSELoss()
    criterion_d = nn.CosineSimilarity(dim=1, eps=1e-6)

    vsum_params = []
    base_params = []

    for name, param in model.named_parameters():
        if "base" not in name:
            vsum_params.append(param)
        elif "mixed_5" in name and "base" in name:
            base_params.append(param)

    if args.optimizer == "adam":
        if args.finetune:
            optimizer = torch.optim.Adam(
                [
                    {"params": base_params, "lr": args.lrs},
                    {"params": vsum_params, "lr": args.lrv},
                ],
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), args.lrv, weight_decay=args.weight_decay
            )
    elif args.optimizer == "sgd":
        if args.finetune:
            optimizer = torch.optim.SGD(
                [
                    {"params": base_params, "lr": args.lrs},
                    {"params": vsum_params, "lr": args.lrv},
                ],
                momentum=args.momemtum,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                args.lrv,
                momentum=args.momemtum,
                weight_decay=args.weight_decay,
            )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=1.0)
    checkpoint_dir = os.path.join(
        os.path.dirname(__file__), args.checkpoint_dir, args.log_name
    )

    if args.checkpoint_dir != "" and args.rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    total_batch_size = args.world_size * args.batch_size
    log(
        "Starting training loop for rank: {}, total batch size: {}".format(
            args.rank, total_batch_size
        ),
        args,
    )


def create_model(args):
    if args.model_type == 1:
        model = s3dg.VSum(
            args.num_class,
            space_to_depth=False,
            word2vec_path=args.word2vec_path,
            init=args.weight_init,
            enc_layers=args.enc_layers,
            heads=args.heads,
            dropout=args.dropout)
    else:
        model = s3dg.VSum_MLP(
            args.num_class,
            space_to_depth=False,
            word2vec_path=args.word2vec_path,
            init=args.weight_init,
            dropout=args.dropout)

    return model


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
