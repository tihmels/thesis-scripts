RAI_TOPIC_PREFIX = 'tensor:topic:'
RAI_SHOT_PREFIX = 'tensor:shot:'
RAI_TEXT_PREFIX = 'tensor:text:'
RAI_STORY_PREFIX = 'tensor:story:'
RAI_M5C_PREFIX = 'tensor:m5c:'
RAI_VIS_PREFIX = 'tensor:vis:'


def get_topic_key(pk: str):
    return RAI_TOPIC_PREFIX + pk


def get_shot_key(pk: str):
    return RAI_SHOT_PREFIX + pk


def get_text_key(pk: str):
    return RAI_TEXT_PREFIX + pk


def get_story_key(pk: str):
    return RAI_STORY_PREFIX + pk


def get_m5c_key(pk: str):
    return RAI_M5C_PREFIX + pk


def get_vis_key(pk: str):
    return RAI_VIS_PREFIX + pk
