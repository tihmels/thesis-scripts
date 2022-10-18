import argparse
import re
from datetime import datetime
from pathlib import Path
from xml.dom.minidom import parse

from common.VideoData import get_date_time, VideoData, get_xml_transcript
from common.constants import TV_FILENAME_RE

parser = argparse.ArgumentParser('Video frame extraction using ffmpeg')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help="TS video files")
parser.add_argument('--overwrite', action='store_false', dest='skip_existing',
                    help='Re-extracts frames for all videos')

trigger_sentence = "Guten Abend, ich begrüße sie zur tagesschau."


def extract_captions(dom):
    spans = dom.getElementsByTagName("tt:span")

    text = ' '.join([span.firstChild.nodeValue for span in spans if span.getAttribute('style') == 'textWhite'])

    return text.strip()


def get_time(timestamp):
    return datetime.strptime(timestamp.partition('.')[0], '%H:%M:%S').time()


def parse_xml(dom):
    body = dom.getElementsByTagName("tt:div")[0]

    sections = body.getElementsByTagName("tt:p")

    intervals = [(get_time(section.getAttribute('begin')),
                  get_time(section.getAttribute('end'))) for section in sections]
    captions = [extract_captions(section) for section in sections]

    return intervals, captions


def process_video(vd: VideoData):
    transcript = get_xml_transcript(vd)

    dom = parse(str(transcript))

    intervals, captions = parse_xml(dom)

    for interval, caption in zip(intervals, captions):
        start, end = interval[0].isoformat(), interval[1].isoformat()
        print("[" + start + " - " + end + "]" + " " + caption)

    print()


def was_processed(file):
    return False


def check_requirements(file: Path):
    if re.match(TV_FILENAME_RE, file.name) is None:
        return False

    if not get_xml_transcript(file).is_file():
        print(f'{file.name} has no xml transcript file')
        return False

    return True


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0

    video_files = sorted(video_files, key=get_date_time)

    print(f'Decoding {len(video_files)} videos', end='\n\n')

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vd}')

        process_video(vd)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
