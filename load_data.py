import os
import random

def load_imdb_data(data_path, seed):
    """Load IMDB train or test data.

    Arguments:
        data_path {string} -- path to train/test directory.
    """
    texts = []
    labels = []

    category_labels = [0, 1]

    for i, category in enumerate(['neg', 'pos']):
        l_path = os.path.join(data_path, category)
        files = sorted(os.listdir(l_path))

        for filename in files:
            if filename.endswith('.txt'):
                f_path = os.path.join(l_path, filename)

                with open(f_path, 'r') as f:
                    text = f.read()
                    texts.append(text)

                label = category_labels[i]
                labels.append(label)

    # suffle
    random.seed(seed)
    random.shuffle(texts)
    random.seed(seed)
    random.shuffle(labels)

    return texts, labels

def load_imdb_sentiment_analysis_dataset(data_path, seed=123):
    """Load IMDB user reviews.

    Arguments:
        data_path {string} -- path to data directory.

    Keyword Arguments:
        seed {int} -- seed for randomizer. (default: {123})
    """
    train_path = os.path.join(data_path, 'train')
    train_texts, train_labels = load_imdb_data(train_path, seed)

    test_path = os.path.join(data_path, 'test')
    test_texts, test_labels = load_imdb_data(test_path, seed)

    return (train_texts, train_labels), (test_texts, test_labels)
