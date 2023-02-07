import numpy as np
import pandas as pd

from common.utils import sec_to_time
from database.model import Story, MainVideo, ShortVideo

ts15_videos = MainVideo.find().all()
ts100_videos = ShortVideo.find().all()

ts15_stories = Story.find(Story.type == 'ts15').all()
ts100_stories = Story.find(Story.type == 'ts100').all()

n_ts15_videos = len(ts15_videos)
n_ts100_videos = len(ts100_videos)

n_ts15_stories = len(ts15_stories)
n_ts100_stories = len(ts100_stories)

ts15_story_durations_in_sec = [np.round((story.last_frame_idx - story.first_frame_idx) / 25, 1) for story in
                               ts15_stories]
ts100_story_durations_in_sec = [np.round((story.last_frame_idx - story.first_frame_idx) / 25, 1) for story in
                                ts100_stories]

ts15_total_duration = sec_to_time(sum(ts15_story_durations_in_sec))
ts100_total_duration = sec_to_time(sum(ts100_story_durations_in_sec))

ts15_stories_per_video = [len(video.stories) for video in ts15_videos]
ts100_stories_per_video = [len(video.stories) for video in ts100_videos]

ts15_shots_per_story = [len(story.shots) for story in ts15_stories]
ts100_shots_per_story = [len(story.shots) for story in ts100_stories]

ts15_shot_duration = [np.round((shot.last_frame_idx - shot.first_frame_idx) / 25, 1) for story in ts15_stories for shot
                      in story.shots]
ts100_shot_duration = [np.round((shot.last_frame_idx - shot.first_frame_idx) / 25, 1) for story in ts100_stories for
                       shot in story.shots]

ts15_shot_duration_series = pd.Series(ts15_shot_duration)
ts100_shot_duration_series = pd.Series(ts100_shot_duration)

ts15_story_duration_series = pd.Series(ts15_story_durations_in_sec)
ts100_story_duration_series = pd.Series(ts100_story_durations_in_sec)

ts15_story_per_video_series = pd.Series(ts15_stories_per_video)
ts100_story_per_video_series = pd.Series(ts100_stories_per_video)

ts15_shots_per_story = pd.Series(ts15_shots_per_story)
ts100_shots_per_story = pd.Series(ts100_shots_per_story)

print(f'TS15 Videos: {len(ts15_videos)}')
print(f'TS100 Videos: {len(ts100_videos)}')

print(f'TS15 Stories: {len(ts15_stories)}')
print(f'TS100 Stories: {len(ts100_stories)}')

print(f'ts15 Story Dur: {ts15_total_duration}')
print(f'ts100 Story Dur: {ts100_total_duration}')

print(f'ts15 Shots: {sum([len(story.shots) for story in ts15_stories])}')
print(f'ts100 Shots: {sum([len(story.shots) for story in ts100_stories])}')

print(f'ts15 Shots Duration: {sum([len(story.shots) for story in ts15_stories])}')
print(f'ts100 Shots Duration: {sum([len(story.shots) for story in ts100_stories])}')

print('Story Duration in sec')
print(ts15_story_duration_series.describe())
print(ts100_story_duration_series.describe())

print('Stories per Video')
print(ts15_story_per_video_series.describe())
print(ts100_story_per_video_series.describe())

print('Shots per Story')
print(ts15_shots_per_story.describe())
print(ts100_shots_per_story.describe())

print('Shot Duration')
print(ts15_shot_duration_series.describe())
print(ts100_shot_duration_series.describe())
