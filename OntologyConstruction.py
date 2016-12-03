import SeminarTagging
import os
from os import listdir
from os.path import isfile, join
import re
import nltk
from collections import defaultdict
import random
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
import numpy
from binarytree import tree
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt
from nltk.corpus import stopwords

training_path = "/home/george/nltk_data/corpora/assignment/nlp_training/training/"
untagged_path = "/home/george/nltk_data/corpora/assignment/nlp_untagged/"
test_path_tagged = "/home/george/nltk_data/corpora/assignment/test_tagged/"
test_path_untagged = "/home/george/nltk_data/corpora/assignment/test_untagged/"


# def train():
#
#     word_dict = defaultdict()
#     classes = defaultdict()
#
#     onlyfiles = [f for f in listdir(training_path) if isfile(join(training_path, f))]
#     if ".DS_Store" in onlyfiles:
#         onlyfiles.remove(".DS_Store")
#
#     for file in onlyfiles[:5]:
#         with open(training_path + file, 'r') as f:
#             text = f.read()
#
#         print(word_dict.keys())
#
#         print(text)
#
#         clas, _, type = raw_input("Type: ").partition(",")
#
#         if clas in classes:
#             classes[clas] += [type]
#         else:
#             classes[clas] = [type]
#
#         sentences = text.split("<sentence>")
#
#         for s in sentences:
#             s = re.sub(r'<\w>', "", s)
#             words = nltk.word_tokenize(s)
#             pos_tags = nltk.pos_tag(words)
#             nouns = [x[0] for x in pos_tags if x[1] in {"NN", "NNP", "NNS", "NNPS"}]
#
#             for noun in nouns:
#                 if type in word_dict:
#                     if noun in word_dict[type]:
#                         word_dict[type][noun] += 1
#                     else:
#                         word_dict[type][noun] = 1
#                 else:
#                     word_dict[type] = defaultdict()
#                     word_dict[type][noun] = 1
#
#     print(word_dict)


def get_file_contents(file):
    with open(file, 'r') as f:
        contents = f.read()
    return contents


def get_vocab():

    vocab = set()

    onlyfiles = [f for f in listdir(training_path) if isfile(join(training_path, f))]
    if ".DS_Store" in onlyfiles:
     onlyfiles.remove(".DS_Store")

    for file in onlyfiles:
        with open(training_path + file, 'r') as f:
            text = f.read()

        text = re.sub(r'</?\w>', "", text)
        sentences = nltk.sent_tokenize(text)
        words = [nltk.word_tokenize(sentence) for sentence in sentences]
        words_postag = [nltk.pos_tag(sentence) for sentence in words]
        words = [i[0] for sublist in words_postag for i in sublist if i[1] in {"NN", "NNP", "NNS", "NNPS"}]
        vocab.update(words)

    stop = set(stopwords.words('english'))
    vocab -= stop

    return list(vocab)


def create_vector(text, vocab):

    text = re.sub(r'</?\w*>', "", text)
    sentences = nltk.sent_tokenize(text)
    words = [nltk.word_tokenize(sentence) for sentence in sentences]
    words_postag = [nltk.pos_tag(sentence) for sentence in words]
    words = [i[0] for sublist in words_postag for i in sublist if i[1] in {"NN", "NNP", "NNS", "NNPS"}]
    vector = [1 if x in words else 0 for x in vocab]
    #print(vector)
    return vector


def rand_vector(length):
    vector = []
    while sum(vector) == 0:
        vector = []
        for i in range(0, length):
            vector += [random.random()]
    return vector


def get_mean_vector(vs):
    return numpy.mean(vs, axis=0)


def kmeans(n, vocab_length, vectors):

    mean_vectors = []
    closest = defaultdict()

    for i in range(0, n):
        mean_vectors += [rand_vector(vocab_length)]

    for i in range(0, 10):
        #print "Mean Vectors: {}".format(mean_vectors)
        for mv in mean_vectors:
            closest[tuple(mv)] = []

        for v in vectors:
            min_distance = 100
            min_mean_vector = mean_vectors[0]
            for sv in mean_vectors:
                d = euclidean(v, sv)
                # print "{} - {} - {}".format(v, sv, d)
                if d < min_distance:
                    min_mean_vector = sv
                    min_distance = d

            closest[tuple(min_mean_vector)] += [v]

        new_means = []
        #print closest
        for mv in mean_vectors:
            new_mean = get_mean_vector(closest[tuple(mv)])
            new_means += [new_mean]

        mean_vectors = new_means

        for i in range(0, len(mean_vectors)):
            if type(mean_vectors[i]) == numpy.float64:
                mean_vectors[i] = rand_vector(vocab_length)

    return mean_vectors


def categorize(file, vocab, means):
    text = get_file_contents(training_path + file)
    vector = create_vector(text, vocab)

    min_distance = 100
    min_index = 0
    min_mean_vector = means[0]
    for index, m in enumerate(means):
        d = euclidean(vector, m)
        # print "{} - {} - {}".format(v, sv, d)
        if d < min_distance:
            min_mean_vector = m
            min_distance = d
            min_index = index

    return min_mean_vector, min_index


def create_tree(vocab):

    onlyfiles = [f for f in listdir(training_path) if isfile(join(training_path, f))]
    if ".DS_Store" in onlyfiles:
        onlyfiles.remove(".DS_Store")

    onlyfiles = sorted(onlyfiles)

    vectors = []
    for file in onlyfiles:
        content = get_file_contents(training_path + file)
        vector = create_vector(content, vocab)
        vectors += [vector]

    Z = linkage(vectors, method='single')

    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.show()


def run():

    vocab = get_vocab()
    #print(vocab)

    onlyfiles = [f for f in listdir(training_path) if isfile(join(training_path, f))]
    if ".DS_Store" in onlyfiles:
        onlyfiles.remove(".DS_Store")

    vectors = []

    for file in onlyfiles:
        text = get_file_contents(training_path + file)
        v = create_vector(text, vocab)
        vectors += [v]

    mean_vectors = kmeans(5, len(vocab), vectors)
    return mean_vectors