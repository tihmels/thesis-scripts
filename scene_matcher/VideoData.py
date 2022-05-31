import itertools
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

TV_DATEFORMAT = '%Y%m%d'


class VideoData:
    def __init__(self, path: Path):
        self.id: str = path.stem
        self.path: Path = path
        self.date: datetime = datetime.strptime(path.parent.name, TV_DATEFORMAT)
        self.frame_dir: Path = Path(path.parent, path.name.split('-')[2])
        self.frames: [Path] = sorted(self.frame_dir.glob('frame_*.jpg'))
        self.segments: [(int, int)] = self.read_segments_from_file(Path(self.frame_dir, 'shots.txt'))

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

    @staticmethod
    def read_segments_from_file(file):
        shots = []

        file = open(file, 'r')
        for line in file.readlines():
            first_index, last_index = [int(x.strip(' ')) for x in line.split(' ')]
            shots.append((first_index, last_index))

        return shots

    def __str__(self):
        return "[" + str(self.__dict__['date'].strftime('%Y%m%d')) + "] " + str(
            self.__dict__['path'].id) + ": " + str(
            self.__dict__['n_frames']) + " frames, " + str(len(self.__dict__['segments'])) + " segments"


class VideoType(Enum):
    SUM = 0
    FULL = 1


class VideoStats:
    def __init__(self, vd: VideoData, vt: VideoType, segment_vector: np.array):
        self.id = vd.id
        self.date = vd.date.strftime("%Y%m%d")
        self.type = vt.name

        data = np.array([*vd.segments], dtype=np.int32)
        data = np.column_stack((data, np.array([(s2 - s1 + 1) for s1, s2 in vd.segments], dtype=np.int32)))
        data = np.column_stack((data, segment_vector.astype(np.int32)))

        start_frame_paths = np.array(vd.frames)[[seg[0] for seg in vd.segments]]
        end_frame_paths = np.array(vd.frames)[[seg[1] for seg in vd.segments]]

        segment_center_indices = [np.round((s1 + s2) / 2).astype(int) for s1, s2 in vd.segments]
        center_frame_paths = np.array(vd.frames)[segment_center_indices]

        data = np.column_stack((data, start_frame_paths))
        data = np.column_stack((data, center_frame_paths))
        data = np.column_stack((data, end_frame_paths))

        self.df = pd.DataFrame(data=data,
                               columns=['seg_start', 'seg_end', 'n_frames', 'match', 's_frame', 'c_frame', 'e_frame'])

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


def get_vs_evaluation_df(main_vs: [VideoStats], summary_vs: [VideoStats]):
    main_vs = sorted(main_vs, key=lambda s: s.date)

    data = np.array([s.date for s in main_vs])
    data = np.column_stack((data, [s.id for s in main_vs]))
    data = np.column_stack((data, [get_mm_ss(int(s.n_frames / 25)) for s in main_vs]))
    data = np.column_stack((data, [s.n_frames for s in main_vs]))
    data = np.column_stack((data, [s.n_segments for s in main_vs]))
    data = np.column_stack((data, [s.n_frames_reused for s in main_vs]))
    data = np.column_stack((data, [np.round(s.frames_reused_ratio * 100, 1) for s in main_vs]))
    data = np.column_stack((data, [np.rint(s.n_frames_reused / 25).astype(int) for s in main_vs]))
    data = np.column_stack((data, [s.n_segments_reused for s in main_vs]))
    data = np.column_stack((data, [np.round(s.segments_reused_ratio * 100, 1) for s in main_vs]))

    sum_vs_by_date = dict([(date, list(vs)) for date, vs in
                           itertools.groupby(sorted(summary_vs, key=lambda s: s.date), lambda s: s.date)])

    n_summaries = [len(vs) for vs in sum_vs_by_date.values()]
    data = np.column_stack((data, n_summaries))

    max_n_frames_summary = [max(vs, key=lambda s: s.n_frames_reused) for vs in sum_vs_by_date.values()]
    data = np.column_stack((data, [summary.id for summary in max_n_frames_summary]))

    max_n_frames_reused = [max([video.n_frames_reused for video in videos]) for videos in sum_vs_by_date.values()]
    data = np.column_stack((data, max_n_frames_reused))
    data = np.column_stack((data, [int(n_frames / 25) for n_frames in max_n_frames_reused]))

    max_n_seg_summary = [max(vs, key=lambda s: s.n_segments_reused) for vs in sum_vs_by_date.values()]
    data = np.column_stack((data, [summary.id for summary in max_n_seg_summary]))

    max_n_segments_reused = [max([video.n_segments_reused for video in videos]) for videos in sum_vs_by_date.values()]
    data = np.column_stack((data, max_n_segments_reused))

    df = pd.DataFrame(data=data,
                      columns=['date', 'main_video',
                               'dur (mm:ss)', 'n_frames', 'n_segments',
                               'n_frames_ru', 'n_frames_ru_perc', 'seconds_ru',
                               'n_segments_ru', 'n_segments_ru_perc', 'n_summaries', 'max_n_frames_summary',
                               'max_n_frames_ru', 'sum_ss_ru', 'max_n_segments_summary',
                               'max_n_segments_ru'])

    return df
