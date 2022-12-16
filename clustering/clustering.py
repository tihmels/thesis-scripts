# !/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
from collections import defaultdict

import hdbscan
import numpy as np
import pandas as pd
import random
import sys
import umap
from hyperopt import Trials, partial, fmin, tpe, space_eval, STATUS_OK, hp
# matplotlib.use('TkAgg')
from nltk.corpus import stopwords
from redis_om import Migrator
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import trange

from database import rai
from database.config import get_topic_key
from database.model import MainVideo, TopicCluster, ShortVideo

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


def score_clusters(clusters, prob_threshold=0.05):
    """
    Returns the label count and cost of a given cluster supplied from running hdbscan
    """

    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold) / total_num)

    return label_count, cost


def random_search(embeddings, space, num_evals):
    """
    Randomly search hyperparameter space and limited number of times
    and return a summary of the results
    """

    results = []

    for i in trange(num_evals):
        n_neighbors = random.choice(space['n_neighbors'])
        n_components = random.choice(space['n_components'])
        min_cluster_size = random.choice(space['min_cluster_size'])

        clusters = generate_clusters(embeddings,
                                     n_neighbors=n_neighbors,
                                     n_components=n_components,
                                     min_cluster_size=min_cluster_size,
                                     random_state=space['random_state'])

        label_count, cost = score_clusters(clusters, prob_threshold=0.05)

        results.append([i, n_neighbors, n_components, min_cluster_size,
                        label_count, cost])

    result_df = pd.DataFrame(results, columns=['run_id', 'n_neighbors', 'n_components',
                                               'min_cluster_size', 'label_count', 'cost'])

    return result_df.sort_values(by='cost')


def bayesian_search(embeddings, space, label_lower, label_upper, max_evals=100):
    """
    Perform bayseian search on hyperopt hyperparameter space to minimize objective function
    """

    trials = Trials()
    fmin_objective = partial(objective, embeddings=embeddings, label_lower=label_lower, label_upper=label_upper)
    best = fmin(fmin_objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    best_params = space_eval(space, best)
    print('best:')
    print(best_params)
    print(f"label count: {trials.best_trial['result']['label_count']}")

    best_clusters = generate_clusters(embeddings,
                                      n_neighbors=best_params['n_neighbors'],
                                      n_components=best_params['n_components'],
                                      min_cluster_size=best_params['min_cluster_size'],
                                      random_state=best_params['random_state'])

    return best_params, best_clusters, trials


def objective(params, embeddings, label_lower, label_upper):
    """
    Objective function for hyperopt to minimize, which incorporates constraints
    on the number of clusters we want to identify
    """

    clusters = generate_clusters(embeddings,
                                 n_neighbors=params['n_neighbors'],
                                 n_components=params['n_components'],
                                 min_cluster_size=params['min_cluster_size'],
                                 random_state=params['random_state'])

    label_count, cost = score_clusters(clusters, prob_threshold=0.05)

    # 15% penalty on the cost function if outside the desired range of groups
    if (label_count < label_lower) | (label_count > label_upper):
        penalty = 0.15
    else:
        penalty = 0

    loss = cost + penalty

    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}


def generate_clusters(embeddings,
                      n_neighbors=15,
                      n_components=5,
                      min_dist=0.1,
                      min_cluster_size=15,
                      min_samples=None,
                      random_state=None):
    # n_neighbors: the number of approximate nearest neighbors used to construct the initial high-dimensional graph.
    # It effectively controls how UMAP balances local versus global structure -
    # low values will push UMAP to focus more on local structure by constraining the number of neighboring points considered when analyzing the data in high dimensions,
    # while high values will push UMAP towards representing the big-picture structure while losing fine detail.

    # min_dist, the minimum distance between points in low-dimensional space.
    # This parameter controls how tightly UMAP clumps points together, with low values leading to more tightly packed embeddings.
    # Larger values of min_dist will make UMAP pack points together more loosely, focusing instead on the preservation of the broad topological structure.

    mapper = umap.UMAP(n_neighbors=n_neighbors,
                       n_components=n_components,
                       min_dist=min_dist,
                       metric='cosine',
                       random_state=random_state)

    umap_embeddings = mapper.fit_transform(embeddings)

    clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               metric='euclidean', min_samples=min_samples,
                               cluster_selection_method='leaf', prediction_data=True).fit(umap_embeddings)

    return mapper, clusters


space = {
    "n_neighbors": range(2, 15),
    "n_components": range(2, 5),
    "min_cluster_size": range(10, 25),
    "random_state": 42
}

hspace = {
    "n_neighbors": hp.choice('n_neighbors', range(3, 16)),
    "n_components": hp.choice('n_components', range(3, 16)),
    "min_cluster_size": hp.choice('min_cluster_size', range(12, 20)),
    "random_state": 42
}

label_lower = 30
label_upper = 100
max_evals = 100


def process_stories(ts15_stories, ts100_stories):
    ts15_tensors = [rai.get_tensor(get_topic_key(story.pk)) for story in ts15_stories]

    # best_params_use, best_clusters_use, trials_use = bayesian_search(tensors, space=hspace, label_lower=label_lower,
    #                                                                label_upper=label_upper, max_evals=max_evals)

    mapper, cluster = generate_clusters(ts15_tensors, 20, 30, 0.05, 25, random_state=42)

    # umap_data = umap.UMAP(n_neighbors=30, n_components=2, min_dist=0.0, metric='cosine').fit_transform(ts15_tensors)
    # result = pd.DataFrame(umap_data, columns=['x', 'y'])
    # result['labels'] = cluster.labels_

    ts15_cluster = defaultdict(list)
    for story, label in zip(ts15_stories, cluster.labels_):
        if label != -1:
            ts15_cluster[label].append(story)

    ts100_tensors = [rai.get_tensor(get_topic_key(story.pk)) for story in ts100_stories]
    ts100_embeddings = mapper.transform(ts100_tensors)

    labels, strengths = hdbscan.approximate_predict(cluster, ts100_embeddings)
    data = [(story, label) for idx, (story, label) in enumerate(zip(ts100_stories, labels)) if strengths[idx] > 0.8]

    ts100_cluster = defaultdict(list)
    for story, label in data:
        if label != -1:
            ts100_cluster[label].append(story)

    headlines = [story.headline for story in ts15_stories]

    docs_df = pd.DataFrame(headlines, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False)
    docs_per_topic_agg = docs_per_topic.agg({'Doc': ' '.join})

    tf_idf, count = c_tf_idf(docs_per_topic_agg.Doc.values, m=len(ts15_stories), ngram_range=(1, 2))
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic_agg, n=20)

    TopicCluster.find().delete()

    for label, stories in ts15_cluster.items():
        ts100s = ts100_cluster[label]

        TopicCluster(index=label,
                     keywords=[w[0] for w in top_n_words[label]],
                     n_ts15=len(stories),
                     n_ts100=len(ts100s),
                     ts15s=stories,
                     ts100s=ts100s).save()

    return


def main():
    ts15_videos = MainVideo.find().sort_by('timestamp').all()
    ts100_videos = ShortVideo.find().sort_by('timestamp').all()

    assert len(ts15_videos) > 0, 'No suitable video files have been found.'

    ts15_stories = [story for video in ts15_videos for story in video.stories if
                    rai.tensor_exists(get_topic_key(story.pk))]
    ts100_stories = [story for video in ts100_videos for story in video.stories if
                     rai.tensor_exists(get_topic_key(story.pk))]

    process_stories(ts15_stories, ts100_stories)

    Migrator().run()

    sys.exit()


if __name__ == "__main__":
    main()
