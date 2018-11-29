from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

NGRAM_RANGE = (1, 2)
MIN_DF = 2
TOP_K = 10000

def ngram_vectorize(train_texts, train_labels, val_texts):

    kwargs = {
        'ngram_range': NGRAM_RANGE,
        'analyzer': 'word',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'dtype': 'int32',
        'min_df': MIN_DF
    }

    vectorizer = TfidfVectorizer(**kwargs)
    vectorizer.fit(train_texts)

    x_train = vectorizer.transform(train_texts)
    x_val = vectorizer.transform(val_texts)


    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)

    x_train = selector.transform(x_train)
    x_val = selector.transform(x_val)

    return x_train, x_val