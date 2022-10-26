#!/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import argparse
import re
from datetime import datetime
from pathlib import Path
from xml.dom.minidom import parse

from pandas import DataFrame

from common.VideoData import get_date_time, VideoData, get_xml_transcript_file, is_summary, \
    get_main_transcript_file
from common.constants import TV_FILENAME_RE

parser = argparse.ArgumentParser('Parse Tagesschau Subtitle-XML file')
parser.add_argument('files', type=lambda p: Path(p).resolve(strict=True), nargs='+', help="Tagesschau video file(s)")
parser.add_argument('--overwrite', action='store_false', dest='skip_existing',
                    help='Re-extracts frames for all videos')

trigger_sentence = "Guten Abend, ich begrüße sie zur tagesschau."


def extract_caption_and_color(dom):
    spans = dom.getElementsByTagName("tt:span")

    if len(spans) == 0:
        return '', ''

    text = ' '.join([span.firstChild.nodeValue for span in spans])

    color = spans[0].getAttribute('style')[4:].lower()

    return text.strip(), color


def get_time(timestamp):
    return datetime.strptime(timestamp.partition('.')[0], '%H:%M:%S').time()


def parse_xml(dom):
    body = dom.getElementsByTagName("tt:div")[0]

    sections = body.getElementsByTagName("tt:p")

    intervals = [(get_time(section.getAttribute('begin')),
                  get_time(section.getAttribute('end'))) for section in sections]
    captions = [extract_caption_and_color(section) for section in sections]

    return intervals, captions


def process_video(vd: VideoData):
    xml_transcript = get_xml_transcript_file(vd)

    dom = parse(str(xml_transcript))

    intervals, captions = parse_xml(dom)

    start, end = zip(*intervals)
    caption, color = zip(*captions)

    df = DataFrame(zip(start, end, caption, color), columns=['start', 'end', 'caption', 'color'])

    df.to_csv(get_main_transcript_file(vd), index=False, header=True)


def was_processed(video):
    return get_main_transcript_file(video).is_file()


def check_requirements(video: Path):
    assert not is_summary(video)

    if not re.match(TV_FILENAME_RE, video.name):
        return False

    if not get_xml_transcript_file(video):
        print(f'{video.name} has no XML transcript file')
        return False

    return True


def main(args):
    video_files = {file for file in args.files if check_requirements(file)}

    if args.skip_existing:
        video_files = {file for file in video_files if not was_processed(file)}

    assert len(video_files) > 0

    video_files = sorted(video_files, key=get_date_time)

    for idx, vf in enumerate(video_files):
        vd = VideoData(vf)

        print(f'[{idx + 1}/{len(video_files)}] {vd}')

        process_video(vd)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
