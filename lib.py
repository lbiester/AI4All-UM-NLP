import nltk
import pandas as pd
import numpy as np
from IPython.display import HTML, display
import tabulate
import matplotlib.pyplot as plt
import wordcloud


DEMOGRAPHIC_CSV = 'data/demographic.csv'
HM_CSV = 'data/cleaned_hm.csv'


def load_joined_data():
    # returns a list of dictionaries instead of using pandas
    demographics = pd.read_csv(DEMOGRAPHIC_CSV)
    happy_moments = pd.read_csv(HM_CSV)
    joined = pd.merge(demographics, happy_moments, left_on='wid', right_on='wid')
    aggregated_moments = []
    for row in joined.itertuples():
        aggregated_moments.append({'hm_text': row.cleaned_hm, 'age': row.age, 'country': row.country, 'wid': row.wid,
                                   'gender': row.gender, 'parenthood': row.parenthood, 'marital': row.marital,
                                   'hmid': row.hmid})
    return aggregated_moments


def load_demographics():
    demographics = []
    df = pd.read_csv(DEMOGRAPHIC_CSV)
    for _, row in df.iterrows():
        demographics.append({})
        for column in df.columns:
            demographics[-1][column] = row[column]
    return demographics


def load_happy_moments():
    happy_moments = []
    df = pd.read_csv(HM_CSV)
    for _, row in df.iterrows():
        happy_moments.append({})
        for column in df.columns:
            happy_moments[-1][column] = row[column]
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
