import pandas as pd
from textblob import TextBlob
import fire

def getSentimentPolarity(text):
    """gets sentiment analysis for text string. 

    Returns polarity (range -1 to +1), where +1 is pos sentiment, -1 is neg sentiment.
    """
    blob = TextBlob(text)  # get TextBlob of text string
    pol, obj = blob.sentiment  # get polarity and objectivity from sentiment

    return pol

def getSentimentObjectivity(text):
    """gets sentiment analysis for text string. 

    Returns subjectivity (range 0 to +1) where +1 is more subjective (opinion), and 0 is more objective (fact based)
    """
    blob = TextBlob(text)  # get TextBlob of text string
    pol, obj = blob.sentiment  # get polarity and objectivity from sentiment

    return obj

def main():

    try:
        tweets = pd.read_csv('project-data/cleaned_tweets.csv')  # reading in cleaned tweets from cleanTweets.py
        tweets.dropna(inplace=True)
    except FileNotFoundError:
        print('File not found. Please check project-data folder for cleaned_tweets.csv. Must run cleanTweets.py prior to running this programme.')

    tweets['Polarity'] = tweets.text.map(lambda x: getSentimentPolarity(x))
    tweets['Subjectivity'] = tweets.text.map(lambda x: getSentimentObjectivity(x))

    tweets.to_csv('project-data/cleanedTweetsSentiment.csv',index=False, header=True)
    print('Tweets with sentiment analysis printed to project-data directory.')

if __name__ == "__main__":
    fire.Fire(main)