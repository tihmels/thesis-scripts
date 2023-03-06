import random
import sys
from pathlib import Path

import imagehash as imagehash
import numpy as np
from PIL import Image

from common.utils import flatten, read_images
from database.model import Story


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def main():
    ts15_stories = Story.find(Story.type == 'ts15').all()

    shots = flatten([ts15.shots for ts15 in ts15_stories])
    keyframes = [shot.keyframe for shot in random.sample(shots, 10000)]
    hashes = []

    for frames in chunker(keyframes, 50):
        images = [Image.open(Path('/Users/tihmels/TS/', frame)) for frame in frames]
        hashes.extend([imagehash.dhash(frame, hash_size=16) for frame in images])

        for image in images:
            image.close()

    sim_mat = np.zeros((len(hashes), len(hashes)))

    for (row, col), x in np.ndenumerate(sim_mat):
        sim_mat[row, col] = hashes[col] - hashes[row]

    sim_mat = np.sort(sim_mat, axis=1)

    mean_sim = sim_mat[:, :20].mean(axis=1)

    args = np.argsort(mean_sim)
    lowest_scores = args[:100]

    for idx, kf in enumerate(lowest_scores):
        image = Image.open(Path('/Users/tihmels/TS/', keyframes[kf]))
        image.save(f'/Users/tihmels/Desktop/similar/{idx:02}.jpg')
        image.close()


if __name__ == "__main__":
    main()

    sys.exit()
