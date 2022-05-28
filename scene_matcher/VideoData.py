from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd


class VideoData:
    def __init__(self, path: Path):
        self._dateformat = '%Y%m%d'

        self.id: str = path.stem
        self.path: Path = path
        self.date: datetime = datetime.strptime(path.parent.name, self._dateformat)
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
        data = np.column_stack((data, np.array([s2 - s1 for s1, s2 in vd.segments], dtype=np.int32)))
        data = np.column_stack((data, segment_vector.astype(np.int32)))

        segment_center_indices = [np.round((s1 + s2) / 2).astype(int) for s1, s2 in vd.segments]
        representative_frames = np.array(vd.frames)[segment_center_indices]

        data = np.column_stack((data, representative_frames))

        self.df = pd.DataFrame(data=data, columns=['seg_start', 'seg_end', 'n_frames', 'match', 'rframe'])
        self.df.index += 1

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
        return sum(self.n_frames_per_segment[np.flatnonzero(self.matched_segments)])

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
        print("-------------------------------------------------")
        print(
            f'{n_reused_segments}/{n_segments} reused segments ({round(self.segments_reused_ratio * 100, 2)} %)')
        print(
            f'{n_reused_frames}/{total_frames} reused frames ({round(self.frames_reused_ratio * 100, 2)} %, {n_reused_frames / 25} sec)')
        print("-------------------------------------------------\n")

    def save_as_csv(self, dir_path: Path, suffix=""):
        if suffix:
            suffix = f'-{suffix}'

        self.df.to_csv(Path(dir_path, self.id + "-" + self.type + suffix + ".csv"))
