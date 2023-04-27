import re
import jieba
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pylab import mpl

import pyLDAvis
import pyLDAvis.lda_model

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
random.seed(0)


def get_stopwords(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
        return stopwords


def get_text(path, mode, sample, n_train=200):
    with open(path + 'inf.txt', encoding='gbk') as _index:
        _index = _index.readline()
        books = _index.strip().split(',')

    all_text = [['', '']]  # content, label
    print("# 正在提取语料")
    for book in tqdm(books):
        with open(path + book + '.txt', encoding='gbk', errors='ignore') as f:
            txt = f.read()
            txt = re.sub('[。!?”](\\n\\u3000\\u3000)+', '注意这里要分段', txt)
            txt = re.sub('[\\sa-zA-Z0-9’!"#$%&\'()（）*+,-./:：;<=>?@，。「」★、…【】《》？“”‘‘！\[\]^_`{|}~]+', '', txt)
            txt = txt.split('注意这里要分段')
            for i, para in enumerate(txt):
                all_text[-1][1] = book
                if len(all_text[-1][0]) < 1500:
                    all_text[-1][0] += para
                else:
                    all_text.append([para, book])
            if len(all_text[-1][0]) < 1500:
                all_text.pop(-1)
    random.shuffle(all_text)

    if sample == 'average':
        count = [0] * 16
        sample_text = []
        for text in all_text:
            label_id = books.index(text[1])
            if len(sample_text) >= n_train:
                random.shuffle(sample_text)
                break
            if count[label_id] < 13:
                count[label_id] += 1
                sample_text.append(text)
    else:
        sample_text = all_text[0: n_train]
        count = [0] * 16
        for text in sample_text:
            label_id = books.index(text[1])
            count[label_id] += 1
    print('每本书采样的段落数量：', count)
    print(len(sample_text))
    plt.figure()
    plt.bar([book[0] for book in books], count)

    if mode == 'char':
        text = [" ".join(x[0]) for x in sample_text]
    else:
        text = [" ".join(jieba.cut(x[0])) for x in sample_text]
    label = [x[1] for x in sample_text]

    return text, label, books


def plot_top_words(model, feature_names, n_top_words, title):
    plt.figure()
    fig, axes = plt.subplots(4, 4, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def get_best_components(tf_vectorizer: CountVectorizer, text, test):
    print("# 正在筛选最佳主题数")
    perplexity = []
    for n_components in tqdm(range(1, 17)):
        tf = tf_vectorizer.fit_transform(text)
        tf_test = tf_vectorizer.transform(test)
        lda = LatentDirichletAllocation(
            n_components=n_components,
            max_iter=100,
            learning_method="online",
            learning_offset=50.0,
            random_state=0,
        )
        lda.fit(tf)
        perplexity.append(lda.perplexity(tf_test))
    plt.figure()
    plt.plot(range(1, 17), perplexity)

    n_components = perplexity.index(min(perplexity)) + 1
    print("# 最佳主题数：" + str(n_components))
    return n_components


if __name__ == "__main__":
    stop_file = 'cn_stopwords.txt'
    data_path = 'data/'
    mode = 'char'  # 'word' or 'char'
    sample = 'average'  # 'average' or 'random'
    n_features = 1500
    n_components = 9  # 只有没有执行筛选程序时发挥作用

    stopwords = get_stopwords(stop_file)  # global
    text, label, books = get_text(data_path, mode=mode, sample=sample, n_train=200)

    # # 用困惑度筛选
    # tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, encoding='gbk', analyzer=mode,
    #                                 stop_words=stopwords)
    # test, _, _ = get_text(data_path, mode=mode, sample=sample, n_train=50)
    # n_components = get_best_components(tf_vectorizer, text=text, test=test)
    #
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, encoding='gbk', analyzer=mode, max_features=n_features,
                                    stop_words=stopwords)

    tf = tf_vectorizer.fit_transform(text)
    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=10,
        learning_method="online",
        learning_offset=50.0,
        random_state=0,
    )
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    plot_top_words(lda, tf_feature_names, 10, "Topics in LDA model")
    score = lda.transform(tf)
    test_label = [list(x).index(max(x)) for x in score]
    res = np.zeros((n_components, len(books)))
    for i in range(len(label)):
        j = books.index(label[i])
        res[test_label[i], j] += 1

    plt.figure()
    plt.imshow(res)
    plt.colorbar()
    plt.tight_layout()
    plt.xticks(np.arange(len(books)), labels=books,
               rotation=45, rotation_mode="anchor", ha="right")
    plt.show()

    pic = pyLDAvis.lda_model.prepare(lda, tf, tf_vectorizer)
    pyLDAvis.save_html(pic, mode + '_res/' + mode + '_lda_topic{0:d}.html'.format(n_components))
    pyLDAvis.show(pic, local=False)
