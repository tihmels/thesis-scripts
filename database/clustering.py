# !/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
from collections import defaultdict

import hdbscan
import numpy as np
import pandas as pd
import umap
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from common.utils import flatten
from database import rai
from database.config import RAI_STORY_PREFIX
from database.model import MainVideo, StoryCluster, ShortVideo

german_stop_words = stopwords.words('german')


# https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
# https://towardsdatascience.com/clustering-sentence-embeddings-to-identify-intents-in-short-text-48d22d3bf02e

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in
                   enumerate(labels)}
    return top_n_words


def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                   .Doc
                   .count()
                   .reset_index()
                   .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                   .sort_values("Size", ascending=False))
    return topic_sizes


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words=german_stop_words + ['wegen']).fit(
        documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def generate_clusters(embeddings,
                      n_neighbors=15,
                      n_components=5,
                      min_cluster_size=15,
                      random_state=None):
    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors,
                                 n_components=n_components,
                                 metric='cosine',
                                 random_state=random_state)
                       .fit_transform(embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               metric='euclidean',
                               cluster_selection_method='eom').fit(umap_embeddings)

    return clusters


def process_stories(stories):
    headlines = [story.headline for story in stories]
    tensors = [rai.get_tensor(RAI_STORY_PREFIX + story.pk) for story in stories]

    cluster = generate_clusters(tensors, 14, 4, 8, 42)

    story_cluster = defaultdict(list)
    for story, label in zip(stories, cluster.labels_):
        if label != -1:
            story_cluster[label].append(story)

    docs_df = pd.DataFrame(headlines, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False)
    docs_per_topic_agg = docs_per_topic.agg({'Doc': ' '.join})

    tf_idf, count = c_tf_idf(docs_per_topic_agg.Doc.values, m=len(stories))

    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic_agg, n=20)
    topic_sizes = extract_topic_sizes(docs_df)
    topic_sizes.head(10)

    for cluster, stories in story_cluster.items():
        StoryCluster(keywords=[w[0] for w in top_n_words[cluster]], stories=stories).save()


def main():
    ts100_videos = ShortVideo.find().sort_by('timestamp').all()
    ts15_videos = MainVideo.find().sort_by('timestamp').all()

    videos = ts15_videos + ts100_videos

    assert len(videos) > 0, 'No suitable video files have been found.'

    stories = flatten([video.stories for video in videos])

    process_stories([story for story in stories if rai.tensor_exists(RAI_STORY_PREFIX + story.pk)])


if __name__ == "__main__":
    main()
