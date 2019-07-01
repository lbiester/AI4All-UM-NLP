import pandas as pd
import numpy as np
from IPython.display import HTML, display
import tabulate
import matplotlib.pyplot as plt
import wordcloud


DEMOGRAPHIC_CSV = 'data/demographic.csv'
HM_CSV = 'data/cleaned_hm.csv'


def load_demographics():
    return pd.read_csv(DEMOGRAPHIC_CSV)


def load_demographics_dict():
    demographics = {}
    df = load_demographics()
    for _, row in df.iterrows():
        demographics[row['wid']] = {}
        for column in df.columns:
            if column != 'wid':
                demographics[row['wid']][column] = row[column]
    return demographics


def load_happy_moments():
    return pd.read_csv(HM_CSV)


def load_happy_moments_dict():
    happy_moments = {}
    df = load_happy_moments()
    for _, row in df.iterrows():
        happy_moments[row['hmid']] = {}
        for column in df.columns:
            if column != 'hmid':
                happy_moments[row['hmid']][column] = row[column]
    return happy_moments


def view_demographics(n=10):
    return pd.read_csv(DEMOGRAPHIC_CSV).head(n=n)


def view_happy_moments(n=10):
    return pd.read_csv(HM_CSV).head(n=n)


def print_as_table(demographic_distribution, title):
    print(title)
    table = []
    for category, percent in demographic_distribution.items():
        table.append((category, percent))
    table = sorted(table, key=lambda x: x[1], reverse=True)
    headers = ('Property', 'Percent')
    display(HTML(tabulate.tabulate(table, headers, tablefmt='html')))


def create_word_cloud(word_list, stop=None):
    all_stop = wordcloud.STOPWORDS.union(stop) if stop else wordcloud.STOPWORDS
    if stop:
        wc = wordcloud.WordCloud(
            background_color="white", height=2700, width=3600, stopwords=all_stop).generate(' '.join(word_list))
    else:
        wc = wordcloud.WordCloud(background_color="white", height=2700, width=3600).generate(' '.join(word_list))
    plt.figure(figsize=(14, 8))
    plt.imshow(wc.recolor(colormap=plt.get_cmap('Set2')), interpolation='bilinear')
    plt.axis("off")


def load_glove_embeddings():
    vectors = {}
    with open('data/glove.840B.300d.txt') as f:
        dim = 300
        for line in f:
            split = line.rsplit(maxsplit=dim)
            vectors[split[0]] = np.array([float(val) for val in split[1:]])
    return vectors
