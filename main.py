from load_data import load_imdb_sentiment_analysis_dataset
from explore_data import plot_class_distribution, plot_frequency_distribution_of_ngrams, plot_sample_length_distribution

def main():
    train_data, test_data = load_imdb_sentiment_analysis_dataset('./data/aclImdb')

    # plot_class_distribution(train_data[1])
    plot_frequency_distribution_of_ngrams(train_data[0])
    plot_sample_length_distribution(train_data[0])

if __name__ == "__main__":
    main()