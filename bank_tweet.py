from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
import re
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word
from nltk.util import ngrams
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize
import pymongo
import pymongo_spark

spark = SparkSession \
    .builder \
    .appName("Banks' Tweets") \
    .getOrCreate()

bank_tweets_df = spark.read.format("CSV").option("header","true").load("BankTweets2.csv")
bank_tweets_df.printSchema()

def processRow(row):
    

    tweet = row
    #Lower case
    tweet.lower()
    #Removes unicode strings like "\u002c" and "x96" 
    tweet = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', tweet)       
    tweet = re.sub(r'[^\x00-\x7f]',r'',tweet)
    #convert any url to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert any @Username to "AT_USER"
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub('[\n]+', ' ', tweet)
    #Remove not alphanumeric symbols white spaces
    tweet = re.sub(r'[^\w]', ' ', tweet)
    #Removes hastag in front of a word """
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Remove :( or :)
    tweet = tweet.replace(':)','')
    tweet = tweet.replace(':(','')
    #remove numbers
    tweet = ''.join([i for i in tweet if not i.isdigit()]) 
    #remove multiple exclamation
    tweet = re.sub(r"(\!)\1+", ' ', tweet)
    #remove multiple question marks
    tweet = re.sub(r"(\?)\1+", ' ', tweet)
    #remove multistop
    tweet = re.sub(r"(\.)\1+", ' ', tweet)
    #lemma
    from textblob import Word
    tweet =" ".join([Word(word).lemmatize() for word in tweet.split()])
    #stemmer
    #st = PorterStemmer()
    #tweet=" ".join([st.stem(word) for word in tweet.split()])
    #Removes emoticons from text 
    tweet = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', tweet)
    #trim
    tweet = tweet.strip('\'"')
    row = tweet
    return row

bank_tweets_df.limit(5).show()
bank_tweets_df.registerTempTable("bank_tweets_counts")
bank_sentiments_counts = spark.sql("""
SELECT COUNT(*) as Counts,BankName
FROM 
bank_tweets_counts
WHERE BankName IN ('PNC','American Express','Bank of America','Discover Card','Chase Bank','Fifth Third','Capital One','Citibank')
GROUP BY
BankName
ORDER BY 
Counts DESC
""")
bank_sentiments_counts.show()
bank_tweets_df.registerTempTable("bank_tweets")
bank_sentiments = spark.sql("""
SELECT text,BankName,created
FROM 
bank_tweets
WHERE BankName IN ('PNC','American Express','Bank of America','Discover Card','Chase Bank','Fifth Third','Capital One','Citibank')
""")
new_df = bank_sentiments.toPandas()
new_df.head()
nltk.download('wordnet')
new_df['text'] = new_df['text'].apply(processRow)
new_df = new_df.dropna()
new_df.drop(new_df[new_df['text'] == 'FALSE'].index, inplace=True)
def sentiment_calc(text):
    try:
        return TextBlob(text).sentiment
    except:
        return None
new_df['sentiment'] = new_df['text'].apply(sentiment_calc)
sentiment_series = new_df['sentiment'].tolist()
columns = ['polarity', 'subjectivity']
dfs = pd.DataFrame(sentiment_series, columns=columns)
dfs['sentiment_category'] = ['positive' if score > 0
                             else 'negative' if score < 0
                                 else 'neutral'
                                     for score in dfs['polarity']]
sentiment_score = pd.concat([new_df,dfs], axis=1)
sentiment_score.head()

#Word Tokenize
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
stop_words = set(stopwords.words('english')) 
def get_token(text):
    # Tokenize the text
    tokens = sent_tokenize(text) 

    #Generate tagging for all the tokens using loop
    for i in tokens: 
        words = nltk.word_tokenize(i) 
        words = [w for w in words if not w in stop_words]  
        #  POS-tagger.  
        tags = nltk.pos_tag(words) 
    return tags

sentiment_score['tokens'] = sentiment_score['text'].apply(get_token)

#import libraries
import nltk
from nltk import ne_chunk
from nltk import word_tokenize
nltk.download('words')
nltk.download('maxent_ne_chunker')

def get_entity(text):
    return ne_chunk(nltk.pos_tag(word_tokenize(text)), binary=False)

sentiment_score['entities'] = sentiment_score['text'].apply(get_entity)

import spacy 
import en_core_web_sm
nlp = en_core_web_sm.load()
  
def get_entity_category(text):
    entity = []
    doc = nlp(text)
    for ent in doc.ents: 
        #print(ent.text, ent.start_char, ent.end_char, ent.label_) 
        myTuple = (ent.text, str(ent.start_char), str(ent.end_char), ent.label_)
        entity.append(myTuple)
    return entity
sentiment_score['entity_category'] =  sentiment_score['text'].apply(get_entity_category)

df = spark.createDataFrame(sentiment_score.astype(str),list(sentiment_score.columns))
df.limit(5).show()

df.registerTempTable("tweets_db")
tweets = spark.sql("""
SELECT
text, BankName, created, polarity, subjectivity,
sentiment_category, tokens, entities,
entity_category
FROM
tweets_db
""")
tweets.show()

#tweets.write.parquet("hdfs:///user/wcococow5477/tweets.parquet")

#pymongo_spark.activate()
#on_time_dataframe = spark.read.parquet('hdfs:///user/wcococow5477/tweets.parquet')
#as_dict = on_time_dataframe.rdd.map(lambda row: row.asDict())
#as_dict.saveToMongoDB('mongodb://localhost:27017/bankdb.tweets')
import  pyspark.sql.functions as F
bank_tweets_count  = tweets.groupBy('BankName').count().sort(F.desc('`count`'))
bank_tweets_count = bank_tweets_count.rdd.map(lambda rec: (str(rec[0]) + ',' + str(rec[1])))
bank_tweets_count.coalesce(1).saveAsTextFile('bank_tweets_count')
