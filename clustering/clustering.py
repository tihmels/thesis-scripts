# !/Users/tihmels/miniconda3/envs/thesis-scripts/bin/python -u
import itertools
import logging
import os
import random
import sys
from collections import defaultdict

import hdbscan
import matplotlib
import numpy as np
import pandas as pd
from hyperopt import Trials, partial, fmin, tpe, space_eval, STATUS_OK, hp
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
from nltk.corpus import stopwords
from redis_om import Migrator
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import trange

from common.utils import set_tf_loglevel, topic_text
from database import rai
from database.model import TopicCluster, Story, get_topic_key

set_tf_loglevel(logging.FATAL)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import umap

german_stop_words = stopwords.words('german')


# https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
# https://towardsdatascience.com/clustering-sentence-embeddings-to-identify-intents-in-short-text-48d22d3bf02e

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names_out()
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
    count = CountVectorizer(ngram_range=ngram_range, token_pattern=r"(?u)\b\w\w\w+\b",
                            stop_words=german_stop_words + ['wegen']).fit(
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

        result = generate_clusters(embeddings,
                                   n_neighbors=n_neighbors,
                                   n_components=n_components,
                                   min_cluster_size=min_cluster_size,
                                   random_state=space['random_state'])

        _, clusters, _ = result

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

    _, clusters, _ = generate_clusters(embeddings,
                                       n_neighbors=params['n_neighbors'],
                                       n_components=params['n_components'],
                                       min_cluster_size=params['min_cluster_size'],
                                       random_state=params['random_state'])

    label_count, cost = score_clusters(clusters, prob_threshold=0.05)

    # 15% penalty on the cost function if outside the desired range of groups
    if (label_count < label_lower) | (label_count > label_upper):
        penalty = 0.25
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
                      csm='eom',
                      random_state=None):
    mapper = umap.UMAP(n_neighbors=n_neighbors,
                       n_components=n_components,
                       min_dist=min_dist,
                       metric='cosine',
                       random_state=random_state)

    umap_embeddings = mapper.fit_transform(embeddings)

    clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               metric='euclidean', min_samples=min_samples,
                               cluster_selection_method=csm, prediction_data=True).fit(umap_embeddings)

    n_clusters = dict.fromkeys(set(clusters.labels_), 0)
    for label in clusters.labels_:
        n_clusters[label] = n_clusters[label] + 1

    return mapper, clusters, n_clusters


space = {
    "n_neighbors": range(2, 25),
    "n_components": range(10, 45),
    "min_cluster_size": range(15, 30),
    "min_samples": range(5, 20),
    "random_state": 42
}

hspace = {
    "n_neighbors": hp.choice('n_neighbors', range(3, 20)),
    "n_components": hp.choice('n_components', range(3, 50)),
    "min_cluster_size": hp.choice('min_cluster_size', range(10, 25)),
    "random_state": 42
}

label_lower = 7
label_upper = 20
max_evals = 150


def visualize_clusters(top_n_words, shape=(5, 4)):
    top_n_words.pop(-1)

    fig, axes = plt.subplots(nrows=shape[0], ncols=shape[1], figsize=(10, 25), dpi=105)

    for ax in axes.flatten():
        ax.set_axis_off()

    for idx, (label, n_words) in enumerate(top_n_words.items()):
        ax_idx = np.unravel_index(idx, shape)

        words = list(reversed([w[0] for w in n_words]))
        score = list(reversed([w[1] for w in n_words]))

        ax = axes[ax_idx]

        ax.set_axis_on()
        ax.barh(words, score)

        if idx % 2 == 1:
            ax.set_yticklabels([])  # Hide the left y-axis tick-labels
            ax.set_yticks([])
            ax.invert_xaxis()  # labels read top-to-bottom
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(range(len(words)))
            ax2.set_yticklabels(words)

        # ax.set_title(f'Topic {idx + 1}')
        ax.title.set_size(10)
        # ax.set_xticks([], [])

    fig.tight_layout()


def process_stories(ts15_stories: [Story], ts100_stories: [Story]):
    ts15_texts = [topic_text(story) for story in ts15_stories]
    ts15_texts_tensors = [rai.get_tensor(get_topic_key(story.pk)) for story in ts15_stories]

    ts100_texts_tensors = [rai.get_tensor(get_topic_key(story.pk)) for story in ts100_stories]

    mapper, cluster, n = generate_clusters(ts15_texts_tensors, n_neighbors=13, n_components=5, min_cluster_size=10, random_state=42)

    print(n[-1])
    ts15_cluster = defaultdict(list)
    for story, label in zip(ts15_stories, cluster.labels_):
        if label != -1:
            ts15_cluster[label].append(story)

    docs_df = pd.DataFrame(ts15_texts, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False)
    docs_per_topic_agg = docs_per_topic.agg({'Doc': ' '.join})

    tf_idf, count = c_tf_idf(docs_per_topic_agg.Doc.values, m=len(ts15_stories), ngram_range=(1, 1))
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic_agg, n=5)

    # visualize_clusters(top_n_words, shape=(10, 2))
    # plt.show()

    ts100_embeddings = mapper.transform(ts100_texts_tensors)

    labels, strengths = hdbscan.approximate_predict(cluster, ts100_embeddings)
    data = [(story, label) for idx, (story, label) in enumerate(zip(ts100_stories, labels)) if strengths[idx] > 0.8]

    ts100_cluster = defaultdict(list)
    for story, label in data:
        if label != -1:
            ts100_cluster[label].append(story)

    TopicCluster.find().delete()

    clusters = []

    for label, stories in ts15_cluster.items():
        ts100s = ts100_cluster[label]

        cluster = TopicCluster(index=label,
                               keywords=[w[0] for w in top_n_words[label]],
                               n_ts15=len(stories),
                               n_ts100=len(ts100s),
                               ts15s=stories,
                               ts100s=ts100s).save()

        clusters.append(cluster)

    cluster_to_table(clusters)
    # cluster_videos_to_table(clusters)

    return


def cluster_videos_to_table(clusters):
    for cluster in clusters:
        print('\multicolumn{2}{l}{Cluster ' + str(cluster.index) + '} \\\\ \midrule')
        for ts15, ts100 in itertools.zip_longest(cluster.ts15s, cluster.ts100s,
                                                 fillvalue=type('', (object,), {"headline": ""})()):
            print(f'{ts15.headline} & {ts100.headline} \\\\')


def cluster_to_table(clusters):
    clusters = sorted(clusters, key=lambda c: c.index)
    for cluster in clusters:
        print(f'{cluster.index} & {", ".join(cluster.keywords)} & {cluster.n_ts15} & {cluster.n_ts100} \\\\')
    print(f'& & {sum([cluster.n_ts15 for cluster in clusters])} & {sum([cluster.n_ts100 for cluster in clusters])}')


def main():
    ts15_stories = Story.find(Story.type == 'ts15').all()
    ts100_stories = Story.find(Story.type == 'ts100').all()

    ts15_stories = [story for story in ts15_stories if rai.tensor_exists(get_topic_key(story.pk))]
    ts100_stories = [story for story in ts100_stories if rai.tensor_exists(get_topic_key(story.pk))]

    assert len(ts15_stories) > 0, 'No suitable stories have been found.'

    process_stories(ts15_stories, ts100_stories)

    Migrator().run()

    sys.exit()


if __name__ == "__main__":
    main()
