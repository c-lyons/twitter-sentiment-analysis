import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
import fire

# importing stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

def tweets_clean(tweets):
    """Converts timestamps to datetime and removes special characters and punctuation from tweets. Hyperlinks are also removed and @'s are stored in new col with @ symbols removed.

    Arguments:
        tweets {df} -- dataframe containing tweets

    Returns:
        tweets -- dataframe with cleaned text data and additional features
    """
    tweets['date_len'] = tweets.created_at.map(lambda x: len(x) if isinstance(x, str) else 'NA')
    tweets = tweets[tweets.date_len==19]  # dropping invalid dates (!= 19 chars)
    tweets.drop('date_len', axis=1, inplace=True)  # dropping date len col

    tweets['created_at'] = tweets.created_at.map(lambda x: pd.to_datetime(x))  # converting to datetime

    tweets = tweets[tweets.is_retweet=="false"]  # dropping retweets

    tweets = clean_links(tweets)  # cleaning links from text
    tweets = get_ats(tweets)  # clean/ get @'s' from text
    tweets = clean_punc(tweets)  # removing hyphens/ apost and rejoining
    tweets = clean_special_chars(tweets)  # removing special chars from text

    return tweets

def tweets_prep(tweets):
    """creates new cols using split of text and calculate tweet length and no of words in tweet"""
    tweet_temp = tweets.copy()

    tweet_temp.loc[:, 'words'] = tweet_temp.text.map(lambda x: x.split())
    tweet_temp.loc[:, 'tweet_len'] = tweet_temp.text.map(lambda x: len(x))
    tweet_temp.loc[:, 'tweet_no_words'] = tweet_temp.words.map(lambda x: len(x))

    return tweet_temp

def get_ats(tweets):
    """Extracts @'s and stores in new col"""
    tweets['at_who'] = tweets.text.map(lambda x: [word for word in x.split() if word[0] == '@'])
    tweets['text'] = tweets.text.map(lambda x: [word for word in x.split() if word[0] != '@'])
    tweets['text'] = tweets.text.map(lambda x: ' '.join(x))

    return tweets

def clean_links(tweets):
    """removes hyperlinks from tweets"""
    tweets['text'] = tweets.text.map(lambda x: [word for word in x.split() if 'https' not in word])
    tweets['text'] = tweets.text.map(lambda x: ' '.join(x))

    return tweets

def clean_punc(tweets):
    """removes apostrophes, hyphens and full stops and replaces with no whitespace"""
    tweets['text'] = tweets.text.map(lambda x: re.sub('(?<=[a-zA-Z])’(?=[a-zA-Z])', '', x))
    tweets['text'] = tweets.text.map(lambda x: re.sub('(?<=[a-zA-Z])\'(?=[a-zA-Z])', '', x))
    tweets['text'] = tweets.text.map(lambda x: re.sub('(?<=[a-zA-Z])-(?=[a-zA-Z])', '', x))
    tweets['text'] = tweets.text.map(lambda x: re.sub('(?<=[a-zA-Z])\.(?=[a-zA-Z])', '', x))

    return tweets

def clean_special_chars(tweets):
    """removes all special chars and numbers and replaces with whitespace"""
    tweets['text'] = tweets.text.map(lambda x: re.sub('[^A-Za-z]+', ' ', x))

    return tweets

def get_upper_words(list_of_words):
    """finds all upper case words in text string"""
    for word in list_of_words:
        if word.isupper():
            return word
        else:
            pass

def tweets_since(tweets, since):
    """Filters tweets to date since trump inaugaration and extracts capital words and if tweet contains endorsement."""
    tweet_temp = tweets.copy()

    tweet_temp = tweet_temp[(tweet_temp.created_at>since)] # trump inaug. date
    tweet_temp['capital_words'] = tweet_temp.words.apply(get_upper_words)  # gettng capitalised words from tweets
    tweet_temp['has_endors'] = tweet_temp.text.apply(lambda x: 1 if 'endorsement' in x else 0)  # trumps endorsement tweets flag

    return tweet_temp

def remove_dropwords(tweets):
    """returns list of words in tweet if  word not in dropwords or stopwords"""
    drop_words = ['twitter', '&amp', 'web', 'android', 'i',
                  '@realdonaldtrump', 'https', '&amp;', 'realdonaldtrump',
                  'iphone', 'android', 'amp', 'false', 'rt']

    words = [x for x in tweets.words]
    words_unlisted = [word for word in np.concatenate(words) if word not in stopwords]
    words_unlisted = [word.lower() for word in words_unlisted]
    words_unlisted = [word.split() for word in words_unlisted]
    words_unlisted = [word for word in np.concatenate(words_unlisted)]

    words_cleaned = [word for word in words_unlisted if word not in drop_words]

    return words_cleaned

def remove_dropwords_caps(tweets):
    """returns list of capital words in tweet if  word not in dropwords or stopwords"""
    cap_tweets = tweets.capital_words

    cap_drop_words = ['amp', 'WH', 'EU', 'OANN', 'ICYMI', 'NAFTA', 'MS-13','NYC', 'ABC','CEO', 'OK', 'NRA', 'CIA', 'UK', 'MSNBC',
                     'WSJ', 'ICE', 'GDP', 'FLOTUS', 'VA', 'J', 'NBC', 'I', 'USMCA', 'PM', 'DNC', 'GOP', 'A', 'FBI','CNN',
                     'DC', 'ISIS', 'USA', 'DACA', 'FEMA', 'NFL', 'NATO', 'VP', 'AG', 'FISA', 'WEF18', 'ICYMI-', 'USCG', 'RNC',
                     'DNC', 'NASA', 'AFL', 'RT', 'OANN', 'U', 'P', 'D', 'H', 'G', 'OPEC', 'NSA']

    cap_words = [x for x in cap_tweets if x != None]
    cap_words_unlisted = [word for word in cap_words if word not in stopwords]
    cap_words_unlisted = [word.upper() for word in cap_words_unlisted]

    cap_clean = [word.upper() for word in cap_words_unlisted if word not in cap_drop_words]

    return cap_clean

def get_key_words(tweets, keyWord):
    """Gets tweets containing user specified word. Skips filtering if no word given."""
    if keyWord == None:

        return tweets

    else:

        tweets['text'] = tweets.text.map(lambda x: [x if keyWord.lower() in x.lower().split() else 'NaN'])
        tweets['text'] = tweets.text.map(lambda x: x[0])  # unpacking list

        return tweets[tweets['text']!='NaN']

def main(keyWord=None, since='2017-01-20', getCaps=False):
    """Tweet cleaner for archived Tweets, primarily designed for Trump's Twitter.

    User inputs a csv containing tweets containing headers:
    "source, text, created_at, retweet_count, favorite_count, is_retweet, id_str"

    Returns csv with cleaned tweets where tweets have special characters removed, and extracted features for use in machine learning.
    Returned csv has following headers:

    source --  Author of tweet
    text  -- text of tweet, cleaned of special chars
    created_at  -- timestamp for tweet
    retweet_count  -- number of retweets
    favorite_count  -- number of favourites
    is_retweet  -- boolean indicator for if retweet
    id_str  -- ID string from tweet
    at_who  -- extracted who tweet is @
    words  -- list of words in text
    tweet_len  -- number of chars in tweet
    tweet_no_words  -- number of words in tweet
    capital_words  -- list of capital words found in tweet (if any)
    has_endors  -- binary indicator if word 'endorsement' found in tweet (for specific Trump analysis)

    Keyword Arguments:

    keyWord {str} -- (Optional) Word to filter tweets by. Will select tweets only containing given word. (default: {None})
    since {str} -- Start date for filtering tweets. (default: {'2017-01-20'})
    getCaps {bool} -- Decides whether to print clouds based on words from full tweets or to limit word cloud words to only the words CAPITALIZED RANDOMLY in Trump's tweets.


    Note: Currently latest tweets are limited to May 2020.
    """
    try:
        tweets = pd.read_csv('project-data/all_tweets.csv')
        tweets.dropna(inplace=True)
    except FileNotFoundError:
        print('File not found. Please check project-data folder.')

    # running cleaning/ data prep functions
    tweets = tweets_clean(tweets)
    tweets = tweets_prep(tweets)
    tweets = tweets_since(tweets, since)
    tweets = get_key_words(tweets, keyWord)  # extracting keyWord tweets
    tweets.to_csv('project-data/cleaned_tweets.csv', header=True, index=False)  # printing csv.
    print(f'cleaned_tweets.csv with {keyWord} keyWord in Tweets since {since} has been printed to project-data directory.')

    # cleaned_words_list = remove_dropwords(tweets)  # getting cleaned words list
    # cleaned_caps_list = remove_dropwords_caps(tweets)  # getting list of capital words

    # # printing lists to json files
    # with open("project-data/cleaned_words.json", 'w') as f:
    #     json.dump(cleaned_words_list, f, indent=2)

    # with open("project-data/cleaned_capital_words.json", 'w') as f:
    #     json.dump(cleaned_caps_list, f, indent=2)

if __name__ == "__main__":
    fire.Fire(main)