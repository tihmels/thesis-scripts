#!/Users/tihmels/Scripts/thesis-scripts/conda-env/bin/python -u

import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=lambda p: Path(p).resolve(strict=True))

    group = parser.add_argument_group('action')
    group.add_argument('--frames', action='store_true')
    group.add_argument('--shots', action='store_true')
    group.add_argument('--csv', action='store_true')
    args = parser.parse_args()

    if args.shots:
        shot_files = [file for file in args.dir.rglob('shots.*')]
        [file.unlink() for file in shot_files]

    if args.csv:
        csv_files = [file for file in args.dir.rglob('TV-*.csv')]
        [file.unlink() for file in csv_files]
