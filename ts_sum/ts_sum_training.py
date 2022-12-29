import random
import torch
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
