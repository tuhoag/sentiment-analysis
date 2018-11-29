from load_data import load_imdb_sentiment_analysis_dataset
from explore_data import plot_class_distribution, plot_frequency_distribution_of_ngrams, plot_sample_length_distribution
from vectorize_data import ngram_vectorize
from build_model import mlp_model
from train_ngram_model import train_ngram_model

def main():
    data = load_imdb_sentiment_analysis_dataset('./data/aclImdb')
    train_ngram_model(data=data)

if __name__ == "__main__":
    main()