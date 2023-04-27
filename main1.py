import re
import jieba
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pylab import mpl
from sklearn import svm
from sklearn.metrics import f1_score, precision_score

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
random.seed(0)


def get_stopwords(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
        return stopwords


def get_text(path, mode, sample, n_train, n_test):
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
            if len(all_text[-1][0]) < 1200:
                all_text.pop(-1)
    random.shuffle(all_text)

    if sample == 'average':
        count_train = [0] * 16
        count_test = [0] * 16
        all_text_train = []
        all_text_test = []
        for text in all_text:
            label_id = books.index(text[1])
            if len(all_text_train) >= n_train and len(all_text_test) >= n_test:
                random.shuffle(all_text_train)
                random.shuffle(all_text_test)
                break
            if count_train[label_id] < 13 and sum(count_train) < n_train:
                count_train[label_id] += 1
                all_text_train.append(text)
            else:
                if count_test[label_id] < 7:
                    count_test[label_id] += 1
                    all_text_test.append(text)

    else:
        all_text_train = all_text[0: n_train]
        all_text_test = all_text[n_train: n_train + n_test]
        count_train = [0] * 16
        count_test = [0] * 16
        for text in all_text_train:
            label_id = books.index(text[1])
            count_train[label_id] += 1
        for text in all_text_test:
            label_id = books.index(text[1])
            count_test[label_id] += 1
    print('训练集每本书采样的段落数量：', count_train)
    print(len(all_text_train))
    print('测试集每本书采样的段落数量：', count_test)
    print(len(all_text_test))

    if mode == 'char':
        text_train = [" ".join(x[0]) for x in all_text_train]
        text_test = [" ".join(x[0]) for x in all_text_test]
    else:
        text_train = [" ".join(jieba.cut(x[0])) for x in all_text_train]
        text_test = [" ".join(jieba.cut(x[0])) for x in all_text_test]
    label_train = [x[1] for x in all_text_train]
    label_test = [x[1] for x in all_text_test]

    return text_train, label_train, text_test, label_test, books


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


def get_best_components(tf_vectorizer: CountVectorizer, text):
    print("# 正在筛选最佳主题数")
    perplexity = []
    for n_components in tqdm(range(1, 17)):
        tf = tf_vectorizer.fit_transform(text)
        lda = LatentDirichletAllocation(
            n_components=n_components,
            max_iter=100,
            learning_method="online",
            learning_offset=50.0,
            random_state=0,
        )
        lda.fit(tf)
        perplexity.append(lda.perplexity(tf))
    plt.figure()
    plt.plot(range(1, 17), perplexity)

    n_components = perplexity.index(min(perplexity)) + 1
    print("# 最佳主题数：" + str(n_components))
    return n_components


def train_model(text_train, label_train, text_test, label_test, books, stopwords, mode, n_components, ifplot=False, test=False):
    n_features = 1500
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, encoding='gbk', analyzer=mode, max_features=n_features,
                                    stop_words=stopwords)

    tf = tf_vectorizer.fit_transform(text_train)
    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=100,
        learning_method="online",
        learning_offset=50.0,
        random_state=0,
    )
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    if ifplot:
        plot_top_words(lda, tf_feature_names, 10, "Topics in LDA model")

    train_score = lda.transform(tf)
    test_score = lda.transform(tf_vectorizer.transform(text_test))
    predictor = svm.SVC(C=1, decision_function_shape='ovr', kernel='poly', degree=n_components//3)
    # 进行训练
    _label_train = [books.index(x) for x in label_train]
    predictor.fit(train_score, _label_train)
    train = predictor.predict(train_score)
    precision_train = precision_score(train, _label_train, average='micro')
    print("Topic:{0:d}; Train Precision: {1:.2f}".format(n_components, precision_train))

    # 进行测试
    if test:
        _label_test = [books.index(x) for x in label_test]
        result = predictor.predict(test_score)
        precision_test = precision_score(result, _label_test, average='micro')
        print("Topic:{0:d}; Test Precision: {1:.2f}".format(n_components, precision_test))
        return precision_test
    else:
        return precision_train


if __name__ == "__main__":
    stop_file = 'cn_stopwords.txt'
    data_path = 'data/'
    mode = 'word'  # 'word' or 'char'
    sample = 'average'  # 'average' or 'random'


    stopwords = get_stopwords(stop_file)  # global
    text_train, label_train, text_test, label_test, books = get_text(data_path, mode=mode, sample=sample, n_train=200, n_test=100)


    # 用困惑度筛选
    # tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, encoding='gbk', analyzer=mode,
    #                                 stop_words=stopwords)
    # n_components = get_best_components(tf_vectorizer, text=text_train)

    train_precision = []
    for n_components in range(1, 17):
        train_precision.append(train_model(text_train, label_train, text_test, label_test, books, stopwords, mode, n_components, ifplot=False, test=True))

    n_components = train_precision.index(max(train_precision)) + 1
    train_model(text_train, label_train, text_test, label_test, books, stopwords, mode, n_components, ifplot=True, test=True)


    # tf = tf_vectorizer.fit_transform(text_train)
    # lda = LatentDirichletAllocation(
    #     n_components=n_components,
    #     max_iter=100,
    #     learning_method="online",
    #     learning_offset=50.0,
    #     random_state=0,
    # )
    # lda.fit(tf)
    # tf_feature_names = tf_vectorizer.get_feature_names_out()
    # plot_top_words(lda, tf_feature_names, 10, "Topics in LDA model")
    #
    # train_score = lda.transform(tf)
    # test_score = lda.transform(tf_vectorizer.transform(text_test))
    # predictor = svm.SVC(gamma='scale', C=1, decision_function_shape='ovr', kernel='rbf')
    # # 进行训练
    # label_train = [books.index(x) for x in label_train]
    # label_test = [books.index(x) for x in label_test]
    # predictor.fit(train_score, label_train)
    #
    # train = predictor.predict(train_score)
    # result = predictor.predict(test_score)
    # print("Precision: {0:.2f}".format(precision_score(train, label_train, average='micro')))
    # print("Precision: {0:.2f}".format(precision_score(result, label_test, average='micro')))

