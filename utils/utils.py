import re
import numpy as np
from wordcloud import STOPWORDS
from nltk.corpus import stopwords as SW_NLTK
import nltk

nltk.download('stopwords')

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

def replace_pattern(input_txt, pattern, new_str):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, new_str, input_txt)
    return input_txt


def clean_tweets(lst):
    # remove twitter Return handles (RT @xxx:)
    lst = np.vectorize(remove_pattern)(lst, "RT @[\w]*:")
    # remove twitter handles (@xxx)
    lst = np.vectorize(remove_pattern)(lst, "@[\w]*")
    # remove URL links (httpxxx)
    lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    lst = np.core.defchararray.replace(lst, "[^a-zA-Z#]", " ")
    lst = np.core.defchararray.replace(lst, "’", "'")
    lst = np.core.defchararray.replace(lst, "'s", "")
    lst = np.core.defchararray.replace(lst, "…", "")
    lst = np.core.defchararray.replace(lst, "[0-9]*", "")
    lst = np.core.defchararray.replace(lst, ".", "")
    lst = np.core.defchararray.replace(lst, ",", "")
    lst = np.core.defchararray.replace(lst, "?", "")
    lst = np.core.defchararray.replace(lst, ":", "")
    lst = np.core.defchararray.replace(lst, "!", "")
    lst = np.core.defchararray.replace(lst, "(", "")
    lst = np.core.defchararray.replace(lst, ")", "")
    lst = np.core.defchararray.replace(lst, "[", "")
    lst = np.core.defchararray.replace(lst, "]", "")
    lst = np.core.defchararray.replace(lst, "*", "")
    return lst

def clean_tweets_normal(lst):
    lst_clean = []
    
    for tweet in lst:
        print(tweet)
        tw =remove_pattern(tweet, "RT @[\w]*:")
        tw =remove_pattern(tw, "@[\w]*:")
        tw =remove_pattern(tw, "https?://[A-Za-z0-9./]*")
        print(tw)
#         tw =replace_pattern(tw, "[^a-zA-Z#]", " ")
        tw =replace_pattern(tw, "’", "'")
        tw =replace_pattern(tw, "'s", "")
        lst_clean.append(tw)
    return lst_clean


def update_stopwords(new_stopwords_list):
    # default stopwords in NLTK
    stopwords = set(SW_NLTK.words('english'))
    # default stopwords in Wordcloud
    stopwords = stopwords.union(set(STOPWORDS))
    # add user customized stop words
    stopwords = stopwords.union(set(new_stopwords_list))
    return stopwords

