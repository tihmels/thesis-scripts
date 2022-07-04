import re
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import os
import pandas as pd

from utils.constants import AUDIO_FILENAME_RE
from utils.fs_utils import get_date_time, get_frame_dir, get_shot_file, read_segments_from_file, get_kf_dir, \
    get_audio_dir

TV_DATEFORMAT = '%Y%m%d'


class VideoData:
    def __init__(self, path: Path):
        self.id: str = path.stem
        self.path: Path = path
        self.date: datetime = get_date_time(path)
        self.frame_dir: Path = get_frame_dir(path)
        self.keyframe_dir: Path = get_kf_dir(path)
        self.audio_dir: Path = get_audio_dir(path)
        self.frames: [Path] = sorted(self.frame_dir.glob('frame_*.jpg'))
        self.kfs: [Path] = sorted(self.keyframe_dir.glob('frame_*.jpg'))
        self.segments: [(int, int)] = read_segments_from_file(get_shot_file(path))

    @property
    def n_frames(self):
        return len(self.frames)

    @property
    def n_segments(self):
        return len(self.segments)

    @property
    def timecode(self):
        return self.id.split("-")[2]

    @property
    def date_str(self):
        return self.date.strftime("%Y%m%d")

    def get_keyframe_file(self, idx):
        return Path(self.keyframe_dir, "shot_" + str(idx) + ".jpg")

    def __str__(self):
        return str(self.path.relative_to(self.path.parent.parent))


class VideoType(Enum):
    SUM = 0
    FULL = 1


class VideoStats:
    def __init__(self, vd: VideoData, vt: VideoType, segment_vector: np.array):
        self.id = vd.id
        self.path = vd.path
        self.date = vd.date.strftime("%Y%m%d")
        self.type = vt.name

        data = np.array([*vd.segments], dtype=np.int32)
        data = np.column_stack((data, np.array([(s2 - s1 + 1) for s1, s2 in vd.segments], dtype=np.int32)))

        if vt == VideoType.FULL:
            data = np.column_stack((data, segment_vector.astype(np.int32)))
        else:
            sum_seg_matched, sum_seg_dist = segment_vector
            data = np.column_stack((data, sum_seg_matched.astype(int)))
            data = np.column_stack((data, np.nan_to_num(sum_seg_dist, posinf=-1, neginf=-1).astype(int)))

        start_frame_paths = np.array(vd.frames)[[seg[0] for seg in vd.segments]]
        end_frame_paths = np.array(vd.frames)[[seg[1] for seg in vd.segments]]

        segment_center_indices = [np.round((s1 + s2) / 2).astype(int) for s1, s2 in vd.segments]
        center_frame_paths = np.array(vd.frames)[segment_center_indices]

        data = np.column_stack((data, start_frame_paths))
        data = np.column_stack((data, center_frame_paths))
        data = np.column_stack((data, end_frame_paths))

        if vt == VideoType.FULL:
            self.df = pd.DataFrame(data=data,
                                   columns=['seg_start', 'seg_end', 'n_frames', 'match', 's_frame', 'c_frame',
                                            'e_frame'])
        else:
            self.df = pd.DataFrame(data=data,
                                   columns=['seg_start', 'seg_end', 'n_frames', 'match', 'conf', 's_frame', 'c_frame',
                                            'e_frame'])

    @property
    def n_segments(self):
        return self.df.shape[0]

    @property
    def n_frames(self):
        return sum(self.n_frames_per_segment)

    @property
    def n_frames_per_segment(self):
        return self.df.iloc[:, 2]

    @property
    def matched_segments(self):
        return self.df.iloc[:, 3]

    @property
    def n_segments_reused(self):
        return len(np.flatnonzero(self.matched_segments))

    @property
    def n_frames_reused(self):
        matched_segments_indices = np.flatnonzero(self.matched_segments)
        return sum(self.n_frames_per_segment[matched_segments_indices])

    @property
    def segments_reused_ratio(self):
        return self.n_segments_reused / self.n_segments

    @property
    def frames_reused_ratio(self):
        return self.n_frames_reused / self.n_frames

    def print(self):
        n_segments = self.n_segments
        n_reused_segments = self.n_segments_reused
        total_frames = self.n_frames
        n_reused_frames = self.n_frames_reused

        print(f'Statistics for {self.id} ({self.type})')
        print("----------------------------------------------------")
        print(
            f'{n_reused_segments}/{n_segments} reused segments ({round(self.segments_reused_ratio * 100, 2)} %)')
        print(
            f'{n_reused_frames}/{total_frames} reused frames ({round(self.frames_reused_ratio * 100, 2)} %, {n_reused_frames / 25} sec)')
        print("----------------------------------------------------\n")

    def save_as_csv(self, dir_path: Path, suffix=""):
        if suffix:
            suffix = f'-{suffix}'

        csv_df = self.df.copy(deep=True)
        csv_df.index += 1

        if not dir_path.exists():
            dir_path.mkdir(parents=True)

        csv_df.to_csv(Path(dir_path, self.id + "-" + self.type + suffix + ".csv"))


def get_mm_ss(seconds):
    return time.strftime("%M:%S", time.gmtime(seconds))


def flatten(t):
    return [item for sublist in t for item in sublist]


def get_binary_frame_vector(vs: VideoStats):
    return flatten(
        [[value] * n for value, n in zip(np.where(vs.matched_segments == 0, 0, 1), vs.n_frames_per_segment)])


def get_frame_vector(vs: VideoStats):
    return flatten([[value] * n for value, n in zip(vs.matched_segments, vs.n_frames_per_segment)])


def get_vs_evaluation_df(main_vs: VideoStats, summary_vs: [VideoStats]):
    data = np.array(main_vs.date)
    data = np.column_stack((data, main_vs.id))
    data = np.column_stack((data, get_mm_ss(int(main_vs.n_frames / 25))))
    data = np.column_stack((data, main_vs.n_frames))
    data = np.column_stack((data, main_vs.n_segments))
    data = np.column_stack((data, main_vs.n_frames_reused))
    data = np.column_stack((data, np.round(main_vs.frames_reused_ratio * 100, 1)))
    data = np.column_stack((data, np.rint(main_vs.n_frames_reused / 25).astype(int)))
    data = np.column_stack((data, main_vs.n_segments_reused))
    data = np.column_stack((data, np.round(main_vs.segments_reused_ratio * 100, 1)))

    data = np.column_stack((data, len(summary_vs)))

    max_n_frames_summary = max(summary_vs, key=lambda s: s.frames_reused_ratio)
    data = np.column_stack((data, max_n_frames_summary.id))  # max_n_frames_summary
    data = np.column_stack((data, np.round(max_n_frames_summary.frames_reused_ratio * 100, 1)))  # max_n_frames_ru
    data = np.column_stack((data, int(max_n_frames_summary.n_frames_reused / 25)))  # sum_ss_ru

    max_n_seg_summary = max(summary_vs, key=lambda s: s.segments_reused_ratio)  # max_n_segments_summary
    data = np.column_stack((data, max_n_seg_summary.id))
    data = np.column_stack((data, np.round(max_n_seg_summary.segments_reused_ratio * 100, 1)))

    df = pd.DataFrame(data=data,
                      columns=['date', 'main_video',
                               'dur (mm:ss)', 'n_frames', 'n_segments',
                               'n_frames_ru', 'n_frames_ru_perc', 'seconds_ru',
                               'n_segments_ru', 'n_segments_ru_perc', 'n_summaries', 'max_n_frames_summary',
                               'ru_frames_perc', 'sum_ss_ru', 'max_n_segments_summary',
                               'ru_segments_perc'])

    return df
