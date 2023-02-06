import numpy as np

from common.utils import frame_idx_to_sec
from database.model import Story, MainVideo, ShortVideo, Shot

ts15_videos = MainVideo.find().all()
ts100_video = ShortVideo.find().all()

ts15_stories = Story.find(Story.type == 'ts15').all()
ts100_stories = Story.find(Story.type == 'ts100').all()

n_ts15_videos = len(ts15_videos)
n_ts100_videos = len(ts100_video)

n_ts15_stories = len(ts15_stories)
n_ts100_stories = len(ts100_stories)

ts15_avg_stories_per_video = np.round(n_ts15_stories / n_ts15_videos, 1)
ts100_avg_stories_per_video = np.round(n_ts100_stories / n_ts100_videos, 1)

ts15_avg_shots_per_story = np.round(np.mean([len(story.shots) for story in ts15_stories]), 1)
ts100_avg_shots_per_story = np.round(np.mean([len(story.shots) for story in ts100_stories]), 1)

ts15_avg_story_duration = np.mean([(story.last_frame_idx - story.first_frame_idx) / 25 for story in ts15_stories])
ts100_avg_story_duration = np.mean([(story.last_frame_idx - story.first_frame_idx) / 25 for story in ts100_stories])

print()
