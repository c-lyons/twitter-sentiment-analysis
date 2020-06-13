import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import fire

analyser = SentimentIntensityAnalyzer()  # initializing analyser

def getSentimentScores(text):
    """Gets sentiment analysis scoring for text string.

    Returns polarity scores for a string.

    The Positive, Negative and Neutral scores represent the proportion of text that falls in these categories.

    The Compound score is a metric that calculates the sum of all the lexicon ratings which have been normalized between -1 (most extreme negative) and +1 (most extreme positive).
    """
    score = analyser.polarity_scores(text)
    return score

def main():

    try:
        tweets = pd.read_csv('project-data/cleaned_tweets.csv')  # reading in cleaned tweets from cleanTweets.py
        tweets.dropna(inplace=True)
    except FileNotFoundError:
        print('File not found. Please check project-data folder for cleaned_tweets.csv. Must run cleanTweets.py prior to running this programme.')

    tweets['score_neg'] = tweets.text.map(lambda x: getSentimentScores(x)['neg'])  # get negativity score
    tweets['score_neu'] = tweets.text.map(lambda x: getSentimentScores(x)['neu'])  # get neutrality score
    tweets['score_pos'] = tweets.text.map(lambda x: getSentimentScores(x)['pos'])  # get positivity score
    tweets['score_compound'] = tweets.text.map(lambda x: getSentimentScores(x)['compound'])  # get compound score

    tweets.to_csv('project-data/cleanedTweetsSentiment.csv',index=False, header=True)
    print('Tweets with sentiment analysis printed to project-data directory.')

if __name__ == "__main__":
    fire.Fire(main)