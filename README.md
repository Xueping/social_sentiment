# Outputs Structure

There are two topics under outputs folder, such as, 'Face Masks' and 'Mental Health'.

Under each topic folder, there are two sub folders: 'weekly_incremental_report_data' and 'weekly_incremental_tweets_data'.

CSV files under 'weekly_incremental_report_data' are used to dashboard. 
CSV files under 'weekly_incremental_tweets_data' are used for exploring detailed data.

Two zip files can be downloaded to use.

# Codes for Word Cloud and Topic Modelling
1. Tweets cleaning: utils/utils.py
2. Topic modelling and word frequency: utils/topic_model.py
3. How to use: utils/MentalHealth_demo_WF_TopicModel.py

# Stop words
These stop words can be found in MentalHealth_demo_WF_TopicModel.py.
~~~
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
~~~
You add your stop words in the list.
