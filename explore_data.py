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