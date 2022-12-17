from common.utils import frame_idx_to_sec
from database.model import Story

stories = Story.find(Story.type == 'ts15').all()

for story in stories:

    if len(story.shots) == 0:
        print(f'Story {story.pk} has no shots! Please check')
        continue

    has_anchor = story.shots[0].type == 'anchor'

    shots = story.shots[1:] if has_anchor else story.shots

    if len(shots) == 0:
        continue

    if len(shots) < 1:
        print(f'Story {story.pk} has no shots!')
        continue

    total_shot_length = sum([shot.last_frame_idx - shot.first_frame_idx for shot in shots])
    avg_shot_length = total_shot_length / len(shots)

    avg_sec = frame_idx_to_sec(avg_shot_length)
    print(f'{story.pk}: {avg_sec}')

