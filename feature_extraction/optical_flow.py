import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch

from feature_extraction.RAFT.core.raft import RAFT


def frame_preprocess(frame, device):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    return frame


def get_cpu_model(model):
    new_model = OrderedDict()
    # get all layer's names from model
    for name in model:
        # create new name and update new model
        new_name = name[7:]
        new_model[new_name] = model[name]
    return new_model


def inference(args):
    # get the RAFT model
    model = RAFT(args)
    # load pretrained weights
    pretrained_weights = torch.load(args.model)

    save = args.save
    if save:
        if not os.path.exists("demo_frames"):
            os.mkdir("demo_frames")

    if torch.cuda.is_available():
        device = "cuda"
        # parallel between available GPUs
        model = torch.nn.DataParallel(model)
        # load the pretrained weights into model
        model.load_state_dict(pretrained_weights)
        model.to(device)
    else:
        device = "cpu"
        # change key names for CPU runtime
        pretrained_weights = get_cpu_model(pretrained_weights)
        # load the pretrained weights into model
        model.load_state_dict(pretrained_weights)

    # change model's mode to evaluation
    model.eval()

    video_path = args.video
    # capture the video and get the first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame_1 = cap.read()

    # frame preprocessing
    frame_1 = frame_preprocess(frame_1, device)

    counter = 0
    with torch.no_grad():
        while True:
            # read the next frame
            ret, frame_2 = cap.read()
            if not ret:
                break
            # preprocessing
            frame_2 = frame_preprocess(frame_2, device)
            # predict the flow
            flow_low, flow_up = model(frame_1, frame_2, iters=args.iters, test_mode=True)
            # transpose the flow output and convert it into numpy array
            ret = vizualize_flow(frame_1, flow_up, save, counter)
            if not ret:
                break
            frame_1 = frame_2
            counter += 1


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--video", type=str, default="./videos/car.mp4")
    parser.add_argument("--save", action="store_true", help="save demo frames")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )

    args = parser.parse_args()
    inference(args)


if __name__ == "__main__":
    main()
