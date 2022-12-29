RAI_TOPIC_PREFIX = 'tensor:topic:'
RAI_SHOT_PREFIX = 'tensor:shot:'
RAI_TEXT_PREFIX = 'tensor:mil-nce:text:'
RAI_STORY_PREFIX = 'tensor:story:'
RAI_M5C_PREFIX = 'tensor:mil-nce:m5c:'
RAI_VIS_PREFIX = 'tensor:mil-nce:vis:'
RAI_PSEUDO_SUM_PREFIX = 'pseudo:summary:'
RAI_PSEUDO_SCORE_PREFIX = 'pseudo:scores:'


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


def get_sum_key(pk: str):
    return RAI_PSEUDO_SUM_PREFIX + pk


def get_score_key(pk: str):
    return RAI_PSEUDO_SCORE_PREFIX + pk
