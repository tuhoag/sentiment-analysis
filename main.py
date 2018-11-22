from load_data import load_imdb_sentiment_analysis_dataset
from explore_data import plot_class_distribution

def main():
    train_data, test_data = load_imdb_sentiment_analysis_dataset('./data/aclImdb')

    plot_class_distribution(train_data[1])

if __name__ == "__main__":
    main()