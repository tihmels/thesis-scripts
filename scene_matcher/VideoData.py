from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image


@dataclass
class Scene:
    index: int
    first_frame_index: int
    last_frame_index: int
    first_frame_path: Path
    last_frame_path: Path
    n_frames: int = field(init=False)

    def __post_init__(self):
        self.n_frames = self.last_frame_index - self.first_frame_index

    def load_scene_frames(self) -> (Image, Image):
        return Image.open(self.first_frame_path), Image.open(self.last_frame_path)


@dataclass
class VideoFile:
    path: Path

    id: str = field(init=False)
    name: str = field(init=False)
    date: str = field(init=False)
    frame_path: Path = field(init=False)
    n_frames: int = field(init=False)
    scenes: [Scene] = field(init=False)
    n_scenes: int = field(init=False)

    def get_frame_path(self):
        return Path(self.path.parent, self.id)

    def __post_init__(self):
        self.name = self.path.name
        self.id = self.name.split('-')[2]
        self.date = self.name.split('-')[1]
        self.frame_path = Path(self.path.parent, self.id)

    def check_requirements(self):
        if not self.path.is_file():
            print(f'{self.path} does not exist')
            return False

        if not self.frame_path.is_dir() or not len(list(self.frame_path.glob('*.jpg'))) > 0:
            print(f'{self.frame_path} is not a directory or empty')
            return False

        if not Path(self.frame_path, "scenes.txt").is_file():
            print(f'{self.frame_path} does not contain a scenes.txt')
            return False

        return True

    def load_scene_data(self):
        self.n_frames = len(list(self.frame_path.glob('*.jpg')))

        scenes = []

        file = open(Path(self.frame_path, 'scenes.txt'), 'r')
        lines = file.readlines()

        frames = sorted(self.frame_path.glob('*.jpg'))

        counter = 0
        for line in lines:
            first_index, last_index = [int(x.strip(' ')) for x in line.split(' ')]
            scenes.append(Scene(counter, first_index, last_index, frames[first_index], frames[last_index]))
            counter += 1

        self.n_scenes = len(scenes)
        self.scenes = scenes

    def load_scene_frames(self, scene_index):
        scene = self.scenes[scene_index]
        frames = sorted(self.frame_path.glob('*.jpg'))
        return Image.open(frames[scene.first_index]), Image.open(frames[scene.last_index])
