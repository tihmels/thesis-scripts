import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', type=lambda p: Path(p).resolve(strict=True), nargs='+')
    parser.add_argument('-s', '--skip', action='store_true', help="skip sbd if already exist")
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    frame_dirs = []
    for directory in args.dirs:
        frame_dirs.extend([d for d in sorted(subdirs(directory)) if check_requirements(d, args.skip)])

    assert len(frame_dirs) > 0, f'{args.dirs} does not contain any subdirectories with frame_*.jpg files.'

    print(f'\nVideo Segmentation ({len(frame_dirs)} videos)')


    def callback_handler(res):
        if res is not None and isinstance(res, Path):
            print(f'{res.relative_to(res.parent.parent)} done')


    if args.parallel:

        with mp.Pool(os.cpu_count(), initializer=mute) as pool:
            [pool.apply_async(process_frame_dir, (d,), callback=callback_handler) for d in frame_dirs]

            pool.close()
            pool.join()

    else:
        for d in frame_dirs:
            result = process_frame_dir(d)
            callback_handler(result)
