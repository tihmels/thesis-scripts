# !/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u

import sys
import umap
from bertopic import BERTopic
from redis_om import Migrator
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

from database import rai
from database.model import Story, get_headline_key

sentence_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2", device="cpu")


def process_stories(ts15_stories: [Story], ts100_stories: [Story]):
    mapper = umap.UMAP(n_neighbors=6,
                       n_components=32,
                       min_dist=0.0,
                       metric='cosine',
                       random_state=42)

    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']

    ts15_headlines = [story.headline for story in ts15_stories[:5]]
    ts15_tensors = [rai.get_tensor(get_headline_key(story.pk)) for story in ts15_stories]

    topic_model = BERTopic(
        umap_model=mapper,
        embedding_model=sentence_model,
        language="multilingual",
        min_topic_size=15,
        calculate_probabilities=False,
        low_memory=True,
        verbose=True)

    topics = topic_model.fit_transform(docs[:100])

    print(topic_model.get_topic_info())

    ts100_headlines = [story.headline for story in ts100_stories]
    ts100_tensors = [rai.get_tensor(get_headline_key(story.pk)) for story in ts100_stories]

    print()
    return


def main():
    ts15_stories = Story.find(Story.type == 'ts15').all()
    ts100_stories = Story.find(Story.type == 'ts100').all()

    ts15_stories = [story for story in ts15_stories if rai.tensor_exists(get_headline_key(story.pk))]
    ts100_stories = [story for story in ts100_stories if rai.tensor_exists(get_headline_key(story.pk))]

    assert len(ts15_stories) > 0, 'No suitable stories have been found.'

    process_stories(ts15_stories, ts100_stories)

    Migrator().run()

    sys.exit()


if __name__ == "__main__":
    main()
