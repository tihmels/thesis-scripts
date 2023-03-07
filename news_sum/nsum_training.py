#!/Users/tihmels/Scripts/thesis-scripts/venv/bin/python -u

import os
import random
import time
from argparse import ArgumentParser
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from shutil import rmtree
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from prettytable import PrettyTable
from tqdm import tqdm

from eval_and_log import evaluate_summary
from nsum_utils import Logger, AverageMeter
from video_loader import NewsSumStoryLoader
from vsum import VSum, VSum_MLP

parser = ArgumentParser('Setup RedisAI DB')
parser.add_argument("--seed", default=1, type=int, help="seed for initializing training.")
parser.add_argument("--model_type", "-m", default=1, type=int, help="(1) VSum_Trans (2) VSum_MLP")
parser.add_argument("--optimizer", type=str, default="adam", choices=['adam', 'sgd'], help="opt algorithm")
parser.add_argument("--num_class", type=int, default=512, help="upper epoch limit")
parser.add_argument("--weight_init", type=str, default="uniform", help="CNN weights inits")
parser.add_argument("--dataset_path", type=str, default="/Users/tihmels/TS/")
parser.add_argument("--out_path", type=str, default="/Users/tihmels/Scripts/thesis-scripts/news_sum/out")
parser.add_argument("--log_root_vsum", type=str, default="vsum_output_log")
parser.add_argument("--dropout", "--dropout", default=0.1, type=float, help="Dropout")
parser.add_argument("--fps", type=int, default=8, help="")
parser.add_argument("--heads", "-heads", default=8, type=int, help="number of transformer heads")
parser.add_argument("--cuda", dest="cuda", action="store_true", help="use CUDA")
parser.add_argument("--finetune", dest="finetune", action="store_true", help="finetune S3D")
parser.add_argument("--video_size", type=int, default=224, help="image size")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--step_size", type=int, default=30, help="scheduler step size")
parser.add_argument("--gamma", type=float, default=0.1, help="scheduler gamma")
parser.add_argument("--ngpus", default=0, type=int, help="Number of available gpus")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("--rank", default=0, type=int, help="Rank.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
         "N processes per node, which has N GPUs. This is the "
         "fastest way to use PyTorch for either single node or "
         "multi node data parallel training",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--epochs",
    default=30,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--batch_size_eval", type=int, default=16, help="batch size eval"
)
parser.add_argument(
    "--pin_memory", dest="pin_memory", action="store_true", help="use pin_memory"
)
parser.add_argument("--num_thread_reader", type=int, default=10, help="")
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
    default=240,
    help="number of frames in each video clip",
)
parser.add_argument(
    "--num_frames_per_segment",
    type=int,
    default=16,
    help="number of frames in each segment",
)
parser.add_argument(
    "--log_freq", type=int, default=1, help="Information display frequence"
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
    "--lrs",
    "--learning-rate-s3d",
    default=0.0001,
    type=float,
    metavar="LRS",
    help="initial learning rate",
    dest="lrs",
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
    "--log_root_tboard", type=str, default="vsum_tboard_log", help="log dir root"
)
parser.add_argument(
    "--log_name",
    default="exp",
    help="name of the experiment for checkpoints and logs",
)
parser.add_argument(
    "--pretrain_cnn_path",
    type=str,
    default="/Users/tihmels/Scripts/thesis-scripts/news_sum/pretrained_weights/s3d_howto100m.pth",
    help="",
)
parser.add_argument(
    "--token_to_word_path",
    type=str,
    default="/Users/tihmels/Scripts/thesis-scripts/news_sum/pretrained_weights/s3d_dict.npy",
    help="",
)
parser.add_argument("--verbose", type=int, default=1, help="")


def create_dir(path: Path, rm_if_exist=False):
    if path.is_dir() and rm_if_exist:
        rmtree(path)

    path.mkdir(parents=True, exist_ok=False if rm_if_exist else True)

    return path


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

    logdir = os.path.join(args.out_path, args.log_root_tboard, args.log_name)
    logger = Logger(logdir)

    return logger


def log(output, args):
    print(output)
    with open(
            os.path.join(
                args.out_path, args.log_root_vsum, args.log_name + ".txt"
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


def evaluate(test_loader, model, epoch, tb_logger, loss_fun, args):
    losses = AverageMeter()
    f_scores = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()

    f_scores_ = AverageMeter()
    precisions_ = AverageMeter()
    recalls_ = AverageMeter()

    model.eval()

    table = PrettyTable()
    table.title = "Eval result of epoch {}".format(epoch)
    table.field_names = ["F-score", "Precision", "Recall", "Loss"]
    table.float_format = "1.3"

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            if args.cuda:
                frames = data["video"].cuda().float()
                gt_summary = data["summary"].cuda().view(-1)
                gt_scores = data["scores"].cuda().view(-1)
            else:
                frames = data["video"].float()
                gt_summary = data["summary"].view(-1)
                gt_scores = data["scores"].view(-1)

            embedding, score = model(frames)

            loss = loss_fun(score.view(-1), gt_scores)

            summary_ids = (
                score.detach().cpu().view(-1).topk(int(0.50 * len(gt_summary)))[1]
            )

            summary_ids_ = (
                score.detach().cpu().view(-1).topk(int(torch.count_nonzero(gt_summary == 1)))[1]
            )

            summary = np.zeros(len(gt_summary), dtype=int)
            summary[summary_ids] = 1

            summary_ = np.zeros(len(gt_summary), dtype=int)
            summary_[summary_ids_] = 1

            f_score, precision, recall = evaluate_summary(
                summary, gt_summary.detach().cpu().numpy()
            )

            f_score_, precision_, recall_ = evaluate_summary(
                summary_, gt_summary.detach().cpu().numpy()
            )

            loss = loss.mean()
            losses.update(loss.item(), embedding.shape[0])
            f_scores.update(f_score, embedding.shape[0])
            precisions.update(precision, embedding.shape[0])
            recalls.update(recall, embedding.shape[0])

            f_scores_.update(f_score_, embedding.shape[0])
            precisions_.update(precision_, embedding.shape[0])
            recalls_.update(recall_, embedding.shape[0])

    loss = losses.avg
    f_score = f_scores.avg
    precision = precisions.avg
    recall = recalls.avg

    f_score_ = f_scores_.avg
    precision_ = precisions_.avg
    recall_ = recalls_.avg

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

    log(
        "ALTERNATIVE"
        "F-Score {} \t"
        "Precision {} \t"
        "Recall {} \t".format(
            f_score_, precision_, recall_,
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


def get_params(model):
    base_params = []
    vsum_params = []

    for name, param in model.named_parameters():
        if "base" not in name:
            vsum_params.append(param)
        elif "mixed_5" in name and "base" in name:
            base_params.append(param)

    return base_params, vsum_params


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
    s = time.time()

    for idx, batch in enumerate(train_loader):

        batch_loss = TrainOneBatch(
            model, optimizer, scheduler, batch, criterion, args
        )

        running_loss += batch_loss

        if (idx + 1) % args.log_freq == 0 and args.verbose:
            d = time.time() - s

            log_state(args, dataset, epoch, d, idx, optimizer, running_loss, tb_logger, train_loader)

            running_loss = 0.0


def log_state(args, dataset, epoch, dtime, idx, optimizer, running_loss, tb_logger, train_loader):
    if args.finetune:
        current_lr = optimizer.param_groups[1]["lr"]
    else:
        current_lr = optimizer.param_groups[0]["lr"]

    log(
        "Epoch %d, Elapsed Time: %s, Batch: %s, Epoch status: %.4f,  Training loss: %.4f, Learning rate: %.6f"
        % (
            epoch + 1,
            str(timedelta(seconds=dtime)),
            f'{idx + 1}/{str(int(np.ceil(len(dataset) / args.batch_size)))}',
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


def TrainOneBatch(model, optimizer, scheduler, data, loss_fun, args):
    if args.cuda:
        frames = data["video"].float().cuda(args.gpu, non_blocking=args.pin_memory)
        scores = data["scores"].cuda(args.gpu, non_blocking=args.pin_memory).view(-1)
    else:
        frames = data["video"].float()
        scores = data["scores"].view(-1)

    print(f'Learning rate: {scheduler.get_last_lr()}')

    optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(True):
        embedding, score = model(frames)
        loss = loss_fun(score.view(-1), scores)
        print(f'Predicted Scores: {score.view(-1).detach().cpu().numpy()[::20]}')
        print(f'GT Scores: {scores.detach().cpu().numpy()[::20]}')
        print(f'Loss: {loss.detach().cpu().numpy()[:10]}')

    if args.cuda:
        gradient = torch.ones((loss.shape[0]), dtype=torch.long).cuda(args.gpu, non_blocking=args.pin_memory)
    else:
        gradient = torch.ones((loss.shape[0]), dtype=torch.long)

    loss.backward(gradient=gradient)
    loss = loss.mean()
    optimizer.step()
    scheduler.step()

    return loss.item()


def create_model(args):
    if args.model_type == 1:
        model = VSum(
            args.num_class,
            space_to_depth=True,
            init=args.weight_init,
            enc_layers=args.enc_layers,
            heads=args.heads,
            dropout=args.dropout,
            token_to_word_path=args.token_to_word_path)
    else:
        model = VSum_MLP(
            args.num_class,
            space_to_depth=True,
            init=args.weight_init,
            dropout=args.dropout)

    return model


def main(args):
    if args.verbose:
        print(args)
        print(f'CUDA available: {torch.cuda.is_available()}')
        print(f'Number of GPUs: {torch.cuda.device_count()}')

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('Create Model ...')
    model = create_model(args)

    print('Load pretrained weights ...')
    net_data = torch.load(args.pretrain_cnn_path)
    model.base_model.load_state_dict(net_data)

    for name, param in model.named_parameters():
        if "base" in name:
            param.requires_grad = False
        if "mixed_5" in name and args.finetune:
            param.requires_grad = True

    if args.cuda:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        model = model.cuda(device)

    dataset = NewsSumStoryLoader(
        base_path=args.dataset_path,
        fps=args.fps,
        num_frames=args.num_frames,
        num_frames_per_segment=args.num_frames_per_segment,
        size=args.video_size,
    )

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_thread_reader,
        pin_memory=args.pin_memory  # needs to be true if trained on GPU
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_eval,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_thread_reader,
        pin_memory=args.pin_memory  # needs to be true if trained on GPU
    )

    logger = create_logger(args)

    create_dir(Path(args.out_path, args.log_root_tboard))

    criterion = nn.MSELoss(reduction='none')

    if args.evaluate:
        print("starting evaluation ...")
        evaluate(test_loader, model, -1, logger, criterion, args)
        return

    base_params, vsum_params = get_params(model)

    optimizer = get_optimizer(model, base_params, vsum_params, args)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    checkpoint_dir = Path(args.out_path, args.checkpoint_dir, args.log_name)

    create_dir(checkpoint_dir)

    total_batch_size = args.batch_size

    create_dir(Path(args.out_path, args.log_root_vsum))

    log(
        "Starting training loop with batch size: {}".format(
            total_batch_size), args,
    )

    for epoch in range(args.epochs):

        if (epoch + 1) % 2 == 0:
            evaluate(test_loader, model, epoch, logger, criterion, args)

        start = timer()

        train(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            train_dataset,
            logger,
            args,
        )

        print('Epoch done')

        end = timer()
        print(f'Elapsed Time: {str(timedelta(seconds=end - start))}')

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


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
