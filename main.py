from load_data import load_imdb_sentiment_analysis_dataset
from explore_data import plot_class_distribution, plot_frequency_distribution_of_ngrams, plot_sample_length_distribution
from vectorize_data import ngram_vectorize
from build_model import mlp_model

def main():
    train_data, test_data = load_imdb_sentiment_analysis_dataset('./data/aclImdb')

    x_train, x_val = ngram_vectorize(train_data[0], train_data[1], test_data[0])

    model = mlp_model(
        layers=2,
        units=64,
        dropout_rate=0.2,
        input_shape=x_train.shape[1:],
        num_classes=2
    )
    print(model.summary())
    # plot_class_distribution(train_data[1])
    # plot_frequency_distribution_of_ngrams(train_data[0])
    # plot_sample_length_distribution(train_data[0])

if __name__ == "__main__":
    main()