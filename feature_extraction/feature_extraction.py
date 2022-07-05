#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

from VideoData import VideoData, get_date_time


def calculate_features(vd: VideoData):
    model = resnet18(pretrained=True)
    model.eval()

    layer = model._modules.get('avgpool')

    transform = transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor()
    ])

    frames = [transform(Image.open(frame)) for frame in vd.kfs]

    for frame in frames:
        prediction = model(frames).squeeze(0).softmax(0)
        my_embedding = torch.zeros(512)

        # 4. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        # 5. Attach that function to our selected layer
        h = layer.register_forward_hook(copy_data)
        # 6. Run the model on our transformed image
        model(frame)
        # 7. Detach our copy function from the layer
        h.remove()
        # 8. Return the feature vector
        print(my_embedding.numpy())


def check_requirements(path: Path, skip_existing=False):
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+')
    parser.add_argument('-s', '--skip', action='store_true', help="skip keyframe extraction if already exist")
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    video_files = []

    for file in args.files:
        if file.is_file() and check_requirements(file, args.skip):
            video_files.append(file)
        elif file.is_dir():
            video_files.extend([video for video in file.glob('*.mp4') if check_requirements(video, args.skip)])

    assert len(video_files) > 0

    video_files.sort(key=get_date_time)

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vd}')

        # with h5py.File(get_feature_file(vf), 'w') as h5f:
        #     calculate_features(vd)
