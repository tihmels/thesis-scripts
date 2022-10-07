#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import shutil
from pathlib import Path
from PIL import Image
from imagehash import average_hash

parser = argparse.ArgumentParser()
parser.add_argument('query', type=lambda p: Path(p).resolve(strict=True), nargs='+')


def main(args):
    files = args.query

    for file in files:
        image = Image.open(file)
        hash = average_hash(image, hash_size=12)

        shutil.copy(file, str(hash) + ".jpg")



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
