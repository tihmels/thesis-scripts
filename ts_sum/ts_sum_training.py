from collections import OrderedDict

import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from argparse import ArgumentParser
from prettytable import PrettyTable
from tqdm import tqdm

import s3dg
from ts_sum.evaluate_and_log import evaluate_summary
from ts_sum.ts_sum_utils import Logger, AverageMeter
from ts_sum.video_loader import TVSumStoryLoader

parser = ArgumentParser('Setup RedisAI DB')
parser.add_argument("--seed", default=1, type=int, help="seed for initializing training.")
parser.add_argument("--model_type", "-m", default=1, type=int, help="(1) VSum_Trans (2) VSum_MLP")
parser.add_argument("--optimizer", type=str, default="adam", choices=['adam', 'sgd'], help="opt algorithm")
parser.add_argument("--num_class", type=int, default=512, help="upper epoch limit")
parser.add_argument("--word2vec_path", type=str, default="data/word2vec.pth", help="")
parser.add_argument("--weight_init", type=str, default="uniform", help="CNN weights inits")
parser.add_argument("--dropout", "--dropout", default=0.1, type=float, help="Dropout")
parser.add_argument("--fps", type=int, default=12, help="")
parser.add_argument("--heads", "-heads", default=8, type=int, help="number of transformer heads")
parser.add_argument("--finetune", dest="finetune", action="store_true", help="finetune S3D")
parser.add_argument("--video_size", type=int, default=224, help="image size")
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--rank", default=0, type=int, help="Rank")
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--epochs",
    default=30,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--batch_size_eval", type=int, default=1, help="batch size eval"
)
parser.add_argument(
    "--pin_memory", dest="pin_memory", action="store_true", help="use pin_memory"
)
parser.add_argument(
    "--num_candidates", type=int, default=1, help="num candidates for MILNCE loss"
)
parser.add_argument("--num_thread_reader", type=int, default=10, help="")
parser.add_argument(
    "--enc_layers",
    "-enc_layers",
    default=6,
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
    "--log_freq", type=int, default=3, help="Information display frequence"
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
parser.add_argument(
    "--pretrain_cnn_path",
    type=str,
    default="/Users/tihmels/Scripts/thesis-scripts/ts_sum/pretrained_weights/s3d_howto100m.pth",
    help="",
)
parser.add_argument("--verbose", type=int, default=1, help="")


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

    logdir = os.path.join(args.log_root, args.log_name)
    logger = Logger(logdir)

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)

    return logger


def log(output, args):
    print(output)
    with open(
            os.path.join(
                os.path.dirname(__file__), "vsum_output_log", args.log_name + ".txt"
            ),
            "a",
    ) as f:
        f.write(output + "\n")


def save_checkpoint(state, checkpoint_dir, epoch, n_ckpt=3):
    torch.save(
        state, os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch))
    )
    if epoch - n_ckpt >= 0:
        oldest_ckpt = os.path.join(
            checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch - n_ckpt)
        )
        if os.path.isfile(oldest_ckpt):
            os.remove(oldest_ckpt)


def evaluate(test_loader, model, epoch, tb_logger, loss_fun, args, dataset_name):
    losses = AverageMeter()
    f_scores = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    model.eval()
    if args.rank == 0:
        log("Evaluating on {}".format(dataset_name), args)
        table = PrettyTable()
        table.title = "Eval result of epoch {}".format(epoch)
        table.field_names = ["F-score", "Precision", "Recall", "Loss"]
        table.float_format = "1.3"

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            labels = data["summary"].view(-1)
            video = data["video"].float()
            scores = data["scores"].view(-1)

            embedding, score = model(video)

            if args.rank == 0:
                loss = loss_fun(score.view(-1), scores)

                summary_ids = (
                    score.detach().cpu().view(-1).topk(int(0.50 * len(labels)))[1]
                )

                summary = np.zeros(len(labels))
                summary[summary_ids] = 1

                f_score, precision, recall = evaluate_summary(
                    summary, labels.detach().cpu().numpy()
                )

                loss = loss.mean()
                losses.update(loss.item(), embedding.shape[0])
                f_scores.update(f_score, embedding.shape[0])
                precisions.update(precision, embedding.shape[0])
                recalls.update(recall, embedding.shape[0])

    loss = losses.avg
    f_score = f_scores.avg
    precision = precisions.avg
    recall = recalls.avg

    if args.rank == 0:
        log(
            "Epoch {} \t"
            "F-Score {} \t"
            "Precision {} \t"
            "Recall {} \t"
            "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                epoch, f_score, precision, recall, loss=losses
            ),
            args,
        )
        table.add_row([f_score, precision, recall, loss])
        tqdm.write(str(table))

        if tb_logger is not None:
            # log training data into tensorboard
            logs = OrderedDict()
            logs["Val_IterLoss"] = losses.avg
            logs["F-Score"] = f_scores.avg
            logs["Precision"] = precisions.avg
            logs["Recall"] = recalls.avg

            # how many iterations we have validated
            for key, value in logs.items():
                tb_logger.log_scalar(value, key, epoch)

            tb_logger.flush()

    model.train()


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('Create Model ...')
    model = create_model(args)

    print('Load pretrained weights ...')
    if args.pretrain_cnn_path:
        net_data = torch.load(args.pretrain_cnn_path)
        model.base_model.load_state_dict(net_data)

    for name, param in model.named_parameters():
        if "base" in name:
            param.requires_grad = False
        if "mixed_5" in name and args.finetune:
            param.requires_grad = True

    train_dataset = TVSumStoryLoader(
        fps=args.fps,
        num_frames=args.num_frames,
        num_frames_per_segment=args.num_frames_per_segment,
        size=args.video_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=False
    )

    logger = create_logger(args)

    tb_logdir = os.path.join(args.log_root, args.log_name)
    if args.rank == 0:
        os.makedirs(tb_logdir, exist_ok=True)

    criterion_c = nn.MSELoss(reduction="none")

    vsum_params = []
    base_params = []

    for name, param in model.named_parameters():
        if "base" not in name:
            vsum_params.append(param)
        elif "mixed_5" in name and "base" in name:
            base_params.append(param)

    optimizer = get_optimizer(model, base_params, vsum_params, args)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=1.0)
    checkpoint_dir = os.path.join(
        os.path.dirname(__file__), args.checkpoint_dir, args.log_name
    )

    if args.checkpoint_dir != "" and args.rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    total_batch_size = args.batch_size

    log(
        "Starting training loop for rank: {}, total batch size: {}".format(
            args.rank, total_batch_size
        ),
        args,
    )

    for epoch in range(args.epochs):

        train(
            train_loader,
            model,
            criterion_c,
            optimizer,
            scheduler,
            epoch,
            train_dataset,
            logger,
            args,
        )

        print('Iteration done')

        if args.rank == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                checkpoint_dir,
                epoch + 1,
            )

            print('Checkpoint saved!')


def get_optimizer(model, base_params, vsum_params, args):
    optimizer = args.optimizer

    if optimizer == "adam":
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
    elif optimizer == "sgd":
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
    return optimizer


def train(
        train_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        epoch,
        dataset,
        tb_logger,
        args,
):
    running_loss = 0.0

    for idx, batch in enumerate(train_loader):

        batch_loss = TrainOneBatch(
            model, optimizer, scheduler, batch, criterion
        )

        running_loss += batch_loss

        if (idx + 1) % args.log_freq == 0 and args.verbose and args.rank == 0:
            log_state(args, dataset, epoch, idx, optimizer, running_loss, tb_logger, train_loader)

            running_loss = 0.0


def log_state(args, dataset, epoch, idx, optimizer, running_loss, tb_logger, train_loader):
    if args.finetune:
        current_lr = optimizer.param_groups[1]["lr"]
    else:
        current_lr = optimizer.param_groups[0]["lr"]
    log(
        "Epoch %d, Epoch status: %.4f, Training loss: %.4f, Learning rate: %.6f"
        % (
            epoch + 1,
            args.batch_size * 1 * float(idx) / len(dataset),
            running_loss / args.log_freq,
            current_lr,
        ),
        args,
    )
    # log training data into tensorboard
    if tb_logger is not None:
        logs = OrderedDict()
        logs["Train loss"] = running_loss / args.log_freq
        logs["Learning rate"] = current_lr
        # how many iterations we have trained
        iter_count = epoch * len(train_loader) + idx
        for key, value in logs.items():
            tb_logger.log_scalar(value, key, iter_count)
        tb_logger.flush()


def TrainOneBatch(model, opt, scheduler, data, loss_fun):
    video = data["video"].float()
    scores = data["scores"].view(-1)

    opt.zero_grad()

    with torch.set_grad_enabled(True):
        embedding, score = model(video)
        loss = loss_fun(score.view(-1), scores)

    gradient = torch.ones((loss.shape[0]), dtype=torch.long)
    loss.backward(gradient=gradient)
    loss = loss.mean()
    opt.step()
    scheduler.step()

    return loss.item()


def create_model(args):
    if args.model_type == 1:
        model = s3dg.VSum(
            args.num_class,
            space_to_depth=True,
            word2vec_path=args.word2vec_path,
            init=args.weight_init,
            enc_layers=args.enc_layers,
            heads=args.heads,
            dropout=args.dropout)
    else:
        model = s3dg.VSum_MLP(
            args.num_class,
            space_to_depth=True,
            word2vec_path=args.word2vec_path,
            init=args.weight_init,
            dropout=args.dropout)

    return model


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
