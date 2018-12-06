from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

NGRAM_RANGE = (1, 2)
MIN_DF = 2
TOP_K = 10000
MAX_SEQUENCE_LENGTH = 500

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

def sequence_vectorize(train_texts, val_texts):
    """Vectorize text as sequences of vectors.

    Arguments:
        train_texts {list} -- List of training texts
        val_texts {list} -- List of validation texts
    """
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)

    return x_train, x_val, tokenizer.word_index