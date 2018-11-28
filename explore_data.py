import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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
        word_lens.append(len(text))

    return np.median(word_lens)

def plot_frequency_distribution_of_ngrams(sample_texts,
                                          ngram_range=(1, 2),
                                          num_ngrams=50):
    pass

def plot_sample_length_distribution(sample_texts):
    pass

def main():
    plot_class_distribution([0, 1])

if __name__ == "__main__":
    main()