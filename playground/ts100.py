import random
import sys
from PIL import Image

from common.utils import flatten
from database.model import Story


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def main():
    ts100_stories = Story.find(Story.type == 'ts100').all()

    shots = flatten([ts100.shots for ts100 in ts100_stories])
    keyframes = [shot.keyframe for shot in random.sample(shots, 100)]

    for idx, kf in enumerate(keyframes):
        image = Image.open(kf)
        image.save(f'/Users/tihmels/Desktop/ts100/{idx:02}.jpg')
        image.close()


if __name__ == "__main__":
    main()

    sys.exit()
