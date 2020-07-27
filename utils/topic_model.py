from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from utils import clean_tweets, update_stopwords
from utils.utils import clean_tweets, update_stopwords
# from sentiment.utils import clean_tweets, update_stopwords
import collections
import pandas as pd
import gensim
from gensim import corpora
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
nltk.download('wordnet')
tknzr = TweetTokenizer()


def word_frequency(wd_list, stopwords, top_k=None):

    all_words = ' '.join([text for text in wd_list])
    filtered_words = [word.lower() for word in all_words.split() if word.lower() not in stopwords]
    counted_words = collections.Counter(filtered_words)

    words_counts = {}
    if top_k is None:
        wc = counted_words.most_common()
    else:
        wc = counted_words.most_common(top_k)
    for letter, count in wc:
        words_counts[letter] = count
    return words_counts


# NLTK’s Wordnet to find the meanings of words, synonyms, antonyms, and more.
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


# WordNetLemmatizer to get the root word.
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def prepare_text_for_lda(text, en_stop):
    tokens = tknzr.tokenize(text)
    # filter token whose length is more than 4
    tokens = [token for token in tokens if len(token) > 4]
    # filter the stop words and lowercase token
    tokens = [token.lower() for token in tokens if token.lower() not in en_stop]
    # NLTK’s Wordnet to find the meanings of words, synonyms, antonyms, and more.
    tokens = [get_lemma(token) for token in tokens]
    # get the root word
    tokens = [get_lemma2(token) for token in tokens]

    return tokens


def lda_model(tweets, stop_words, num_topic, num_words):
    text_data = []
    for tweet in tweets:
        tokens = prepare_text_for_lda(tweet, stop_words)
        text_data.append(tokens)

    # build dictionary id2word
    dictionary = corpora.Dictionary(text_data)

    # create corpus, document to bag of words
    corpus = [dictionary.doc2bow(text) for text in text_data]
    # print(corpus)
    ldamodel = gensim.models.ldamodel.LdaModel(corpus,
                                               num_topics=num_topic,
                                               id2word=dictionary,
                                               passes=15)
    topics = ldamodel.print_topics(num_words)
    return topics, dictionary, corpus


if __name__ == '__main__':

    num_topic = 10
    num_words = 10

    analyser = SentimentIntensityAnalyzer()
    file_name = "tweets_trump_wall.csv"
    df_text = pd.read_csv(file_name)
    tweets = clean_tweets(df_text.text)

    # additional stopwords
    new_stopwords = [ '&amp;', '-',  '…',  'one', 'got', 'to…', '...']
    stop_words = update_stopwords(new_stopwords)

    # get word frequency
    word_frq = word_frequency(tweets, stop_words)
    print(word_frq)

    # get topic model
    topics, dictionary, corpus = lda_model(tweets, stop_words, num_topic, num_words)
    for topic in topics:
        print(topic)
    # token to id in dictionary
    print(dictionary.token2id)
    # token_id to document
    print(corpus)


