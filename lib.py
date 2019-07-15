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
        aggregated_moments.append({'cleaned_hm': row.cleaned_hm, 'age': row.age, 'country': row.country, 'wid': row.wid,
                                   'gender': row.gender, 'parenthood': row.parenthood, 'marital': row.marital,
                                   'hmid': row.hmid})
    return aggregated_moments


def load_demographics():
    demographics = []
    df = pd.read_csv(DEMOGRAPHIC_CSV)
    for row in df.itertuples():
        demographics.append({})
        for column in df.columns:
            demographics[-1][column] = row.__getattribute__(column)
    return demographics


def load_happy_moments():
    happy_moments = []
    df = pd.read_csv(HM_CSV)
    for row in df.itertuples():
        happy_moments.append({})
        for column in df.columns:
            happy_moments[-1][column] = row.__getattribute__(column)
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


def create_word_cloud(sentence_list, stop=None):
    text = '\n'.join(sentence_list)
    if stop is not None:
        for word in stop:
            text = text.replace(' ' + word, '')
            text = text.replace(word + ' ', '')

    wc = wordcloud.WordCloud(background_color="white", height=2700, width=3600).generate(text)
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


def create_pie(distribution, title=None):
    labels = distribution.keys()
    sizes = distribution.values()

    colors = plt.cm.viridis(np.linspace(0., 1., 6))

    _, ax = plt.subplots()
    wedges, texts = ax.pie(sizes, startangle=180, labeldistance=1.05, colors=colors)

    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    ax.axis('equal')
    plt.title(title)
    plt.show()
