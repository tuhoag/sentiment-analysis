import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def get_num_classes(labels):
    """Get total number of classes.

    Arguments:
        labels {list} -- label values.
    """
    num_classes = max(labels) + 1
    missing_classes = []

    for l in range(num_classes):
        if l not in labels:
            missing_classes.append(l)

    if len(missing_classes):
        raise ValueError('Missing labels: {missing_classes}'.format(missing_classes=missing_classes))

    if num_classes <= 1:
        raise ValueError('Invalid number of labels: {num_classes}'.format(num_classes=num_classes))

    return num_classes

def plot_class_distribution(labels):
    """Plot class distribution.

    Arguments:
        labels {list} -- list of labels.
    """

    num_classes = get_num_classes(labels)

    counter_map = Counter(labels)
    counts = [ counter_map[i] for i in range(num_classes) ]

    idx = range(num_classes)

    plt.bar(idx, counts)
    plt.show()

def get_num_words_per_sample(sample_texts):
    """Get the number of words per sample.

    Arguments:
        sample_texts {list} -- sample texts.
    """
    word_lens = []

    for text in sample_texts:
        word_lens.append(len(text.split()))

    return np.median(word_lens)

def plot_frequency_distribution_of_ngrams(sample_texts,
                                          ngram_range=(1, 2),
                                          num_ngrams=50):
    """Plot frequency of all ngrams.

    Arguments:
        sample_texts {list} -- list of texts.

    Keyword Arguments:
        ngram_range {tuple} -- range of ngram size to plot (default: {(1, 2)})
        num_ngrams {int} -- number of top frequent ngram to plot (default: {50})
    """

    kwargs = {
        'ngram_range': ngram_range,
        'analyzer': 'word',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'dtype': 'int32'
    }

    vectorizer = CountVectorizer(**kwargs)
    vectorizer.fit(sample_texts)

    vectorized_texts = vectorizer.transform(sample_texts)

    all_ngrams = list(vectorizer.get_feature_names())
    num_ngrams = min(num_ngrams, len(all_ngrams))

    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(zip(all_counts, all_ngrams), key=lambda t: t[0], reverse=True)])
    plot_counts = all_counts[:num_ngrams]
    plot_ngrams = all_ngrams[:num_ngrams]

    idx = np.arange(num_ngrams)
    plt.bar(idx, plot_counts, width=0.6)
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title('Frequency distribution of n-grams')
    plt.xticks(idx, plot_ngrams, rotation=60)
    plt.show()

def plot_sample_length_distribution(sample_texts):
    """Plot text length distribution.

    Arguments:
        sample_texts {list} -- list of texts.
    """

    all_counts = [len(s) for s in sample_texts]
    plt.hist(all_counts, 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()

def main():
    plot_class_distribution([0, 1])

if __name__ == "__main__":
    main()