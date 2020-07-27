import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from datetime import date, timedelta
from scipy.sparse import csr_matrix

from utils.utils import clean_tweets, update_stopwords, clean_tweets_normal
from utils.topic_model import word_frequency, lda_model


if __name__ == '__main__':
    
    #1. Loading existing CSV files
    csv_root = 'outputs/mental_health/weekly_incremental_tweets_data/'

    dicts = {} # store distinct URL
    for _, _, csv_files in os.walk(csv_root):
        csv_files.sort()
        for jf in csv_files:
            print(jf)
            file_path = os.path.join(csv_root, jf)
            df_o = pd.read_csv(file_path, header=0, error_bad_lines=False)
            df_tmp = df_o[['tweet_topic','user_id','user_URL','created_at','URL','tweet_text','sentiment','state','gender','age']]
            for index, row in df_tmp.iterrows():
                dicts[row['URL']] = row

    row_dicts = []
    for row in dicts.values():
        row_dicts.append(row.to_dict())
    df = pd.DataFrame.from_dict(row_dicts) 
    
    # 2. Date processing to change US time to local time
    element =  pd.to_datetime(df['created_at'], format="%Y-%m-%d %H:%M:%S") + pd.to_timedelta(10, unit='h')
    element_time = element.dt.strftime('%Y-%m-%d %H:%M:%S')
    element_date = element.dt.strftime('%Y-%m-%d')
    df['created_at_local'] = element_date
    df['created_at_local_time'] = element_time
    df['created_at_usa'] = df.created_at.str.slice(start=0, stop=10)
    
    # 3. Process All topics
    
    # additional stopwords
    new_stopwords = [ '&amp;', '-',  '…',  'one', 'got', 'to…', '...', 'la', 'de', 'UK', 'India.', 'à', 'un','qui', 'Un', 
                     'du', '»', '«', 'tel', 'Ve,', 'le', 'des', 'sur', 'et','een', 'les', 'si',  'e', '#China', ':' , 'us', 
                     'per', 'new', '1', '2', '3', '4', '5','6', '7','8', '9', '+', '!',  '–',  'https:',  'B"', "'the", 
                     'W', 'l',  'p', 'u', ';' ,     ",",  "a", 'o"', 'ja!', "good'", '12', 'RT', 'sel', 'say', '1)', 
                     '2)', 'op', '#__', '#____', 'se', '15', '&', 'pm', 'tru','#___', '#_', '#', '|', 'el', 'pa', 
                     'SNTE', 'go', 'every', '"', '#COVID2', 'A&amp;E','Trump', 'Donald','10', 'gel', '2020', 'dr', '+ve', 
                     '#dogs', 'gop', '£50', '#gunshot', 'kelli', 'married', 'immigrant', 'americans', 'm&m', 'lol', 'fam', 
                     'god', 'false', 'why/why', '44', '0000000053%','“soft', 'pu', 'nbn', '25th', 'iran', '740000', 'wtf', 
                     '=', '""', "'", '6-feet', 'il', '245', 'plc', 'xxl', 'là', "''", '“', '“”', '4-27', '2500', 'ke',
                    '27', '-19', '”', '*', 'n', '850', '121', '70', '11', '900', '13', '40', '(', '659', '467', '22',
                     'afl', '1800', '978', '789', '14', '1300', '6am', '6pm', '650', '890', '36', '46', 'health', 'mental'
                    ]

    general_topic = 'mental-health-general-australia'
    sub_topics = ['mental-health-availability-australia',
                  'mental-health-recognition-australia',
                  'mental-health-awareness-australia',
                  'mental-health-digital-australia']
    polarity = ['positive', 'negative', 'neutral']

    output_path = 'outputs/demo/'

    num_topic = 5
    num_words = 10

    for sub, topic in enumerate(sub_topics):

        df_topic = df[(df.tweet_topic==general_topic) | (df.tweet_topic==topic)]

        stopwords_low = []
        for sw in new_stopwords:
            stopwords_low.append(sw.lower())
        stop_words = update_stopwords(stopwords_low)

        #get all words
        all_tweets = df_topic.tweet_text
        tweets = clean_tweets(all_tweets)
        # get word frequency
        all_word_frq = word_frequency(tweets, stop_words)

    #---------------------topic modelling for ALL------------------------------------------------------
        # topic model for one type of mental health
        topics, dictionary, corpus = lda_model(tweets, stop_words, num_topic, num_words)
        tpc_ls = []
        all_words = []
        for tpc in topics:
            dicts = dict()
            dicts["Topic"] = 'Topic '+ str(int(tpc[0])+1)
            words = [w.split("*")[1].replace('"', '') for w in tpc[1].split(" + ")]
            for i, w in enumerate(words):
                dicts["word"+str(i+1)] = w
                all_words.append(w)
            tpc_ls.append(dicts)
        pd.DataFrame(tpc_ls).to_csv(output_path+topic+'_topics-All.csv', index=False)

        # get topic word mapping tweets
        uniqueWords=set(all_words)
        tweets_ls = []
        for word in uniqueWords:
            word_tweets = df_topic[(df_topic.tweet_text.str.lower().str.contains(pat = word.lower()))].iloc[0:10]
            word_tweets['keyword'] = word
            tweets_ls.append(word_tweets)
        pd.concat(tweets_ls).to_csv(output_path+topic+'_topics-All-tweets.csv', index=False) 

    #---------------------topic modelling for three types of sentimets------------------------------------------------------    
        for sentiment in polarity:
            sentiment_tweets = df_topic[df_topic.sentiment==sentiment].tweet_text
            tweets = clean_tweets(sentiment_tweets)
            # get word frequency
            sentiment_word_frq = word_frequency(tweets, stop_words)

            #---------------------topic modelling for sentiment------------------------------------------------------
            # topic model for one type of mental health in sentiment
            topics, dictionary, corpus = lda_model(tweets, stop_words, num_topic, num_words)
            tpc_ls = []
            all_words = []
            for tpc in topics:
                dicts = dict()
                dicts["Topic"] = 'Topic '+ str(int(tpc[0])+1)
                words = [w.split("*")[1].replace('"', '') for w in tpc[1].split(" + ")]
                for i, w in enumerate(words):
                    dicts["word"+str(i+1)] = w
                    all_words.append(w)
                tpc_ls.append(dicts)
            pd.DataFrame(tpc_ls).to_csv(output_path+topic+'_topics-'+sentiment+'.csv', index=False)

            uniqueWords=set(all_words)
            tweets_ls = []
            for word in uniqueWords:
                word_tweets = df_topic[(df_topic.sentiment==sentiment) & 
                                            (df_topic.tweet_text.str.lower().str.
                                             contains(pat = word.lower()))].iloc[0:10]
                word_tweets['keyword'] = word
                tweets_ls.append(word_tweets)
            pd.concat(tweets_ls).to_csv(output_path+topic+'_topics-'+sentiment+'-tweets.csv', index=False) 

            #---------------------word cloud for sentiment------------------------------------------------------
            # p(c|w) * log(frequency) For positive:
            pos_dict = {}
            for key in sentiment_word_frq.keys():
                pos_dict[key] = sentiment_word_frq[key]/all_word_frq[key]* np.log(sentiment_word_frq[key])

            pos_dict_sort = {k: v for k, v in sorted(pos_dict.items(), key=lambda item: item[1], reverse=True)}
            # top 100， json file
            json_f = {}
            format_list = []
            common_index = 0
            for index, item in enumerate(pos_dict_sort.items()):
                if index < 100:
                    json_f[item[0]] = int(round(item[1],2)*100)
                    dicts = dict()
                    dicts['keyword'] = item[0]
                    dicts['counts'] = int(round(item[1],2)*100)
                    format_list.append(dicts)

            pd.DataFrame(format_list).to_csv(output_path+topic+'_WF_'+sentiment+'-count.csv', index=False)   

            #---------------------word tweets for sentiment------------------------------------------------------
            tweets_ls = []
            for item in format_list:
                word_tweets = df_topic[(df_topic.sentiment==sentiment) & 
                                            (df_topic.tweet_text.str.lower().str.
                                             contains(pat = item['keyword'].lower()))].iloc[0:10]

                word_tweets['keyword'] = item['keyword']
                tweets_ls.append(word_tweets)
            pd.concat(tweets_ls).to_csv(output_path+topic+'_WF_'+sentiment+'-tweets.csv', index=False) 