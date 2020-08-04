from pyspark.sql import Row, Window
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import functions as F

import pandas as pd
import math

import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.mllib.stat import Statistics

from googletrans import Translator
from textblob import TextBlob

from pyspark.ml.feature import CountVectorizer, StringIndexer, RegexTokenizer,StopWordsRemover
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import SparkSession
# sc = SparkContext()
# spark = SparkSession(sc)
# sqlc = SQLContext(sc)
# print(sc.getConf().getAll())     


def clean_data(train_json_path='RS_v2_2006-03', test_json_path='RS_v2_2006-04', spark=None):
    """Returns train and test set dataframes as a tuple"""

    """"""""""""""""""""
    """STEP 2: importing data"""
    """"""""""""""""""""

    df_train = spark.read.format('json').option('inferSchema', 'false').option('header', 'false').option('sep', ',').load(train_json_path)
    df_test = spark.read.format('json').option('inferSchema', 'false').option('header', 'false').option('sep', ',').load(test_json_path)

    """"""""""""""""""""
    """STEP 3: CLEANING DATA"""
    """"""""""""""""""""

    """SELECTING columns with single or null values in the columns -------------------------------------------"""

    single_value_column_list = []
    for i in df_train.columns:
      b = df_train.select(i).distinct().collect()
      a = [str(row[i]) for row in b]
      if len(a) == 1:
        single_value_column_list.append(i)
        
      elif len(a) == 2 and 'None' in a and ('' in a or '[]' in a):
        single_value_column_list.append(i)
        
      elif len(a) == 2 and '[deleted]' in a and '' in a:
        single_value_column_list.append(i) ## drop_list_

    """SELECTING columns with single or null values in the columns (based on observation) -------------------------------------------"""

    image_embed_columns = ['media_embed', 'secure_media_embed'] #columns with image type embedding

    """SELECTING columns with sparse data -------------------------------------------"""

    threshold_sparsity = 0.80

    remaining_col = set(df_train.columns) - set(image_embed_columns) - set(single_value_column_list)
    sparse_columns = []
    total_rows = df_train.count()
    for i in remaining_col:
      null_count = df_train.where(col(i).isNull()).count()
      null_perc = null_count/total_rows
      if null_perc >= threshold_sparsity:
        sparse_columns.append(i)

    """SELECTING columns with statistically insignificant data -------------------------------------------"""
    """cut columns with overdominant groups"""
    statistical_threshold = 0.90

    remaining_col_2 = set(remaining_col) - set(sparse_columns)
    statistically_insignificant_list = []
    total_rows = df_train.count()
    for i in remaining_col_2:
      a = df_train.groupBy(i).count()
      max_count = a.agg({"count": "max"}).collect()[0]['max(count)']
      if max_count/total_rows >= statistical_threshold:
        statistically_insignificant_list.append(i)

    """SELECTING columns with high correlation based on due to similar values -----------------------------------------"""

    drop_list_high_corr = ['parent_whitelist_status', 'subreddit_id', 'subreddit_name_prefixed', 'permalink', 'id']

    """Combining all columns to drop list"""

    master_list_to_drop = single_value_column_list + image_embed_columns + sparse_columns + statistically_insignificant_list + drop_list_high_corr

    '''Column that has a single value or total null values'''

    drop_list1 = ['archived', 'author_flair_background_color', 'author_flair_css_class', 'author_flair_richtext', 'author_flair_text',
               'contest_mode', 'distinguished', 'edited', 'gilded', 'hidden', 'hide_score', 'is_reddit_media_domain', 'is_self', 'is_video',
               'link_flair_css_class', 'link_flair_richtext', 'link_flair_text', 'link_flair_text_color', 'link_flair_type', 'locked', 'media',
               'media_embed', 'num_crossposts', 'rte_mode', 'secure_media', 'secure_media_embed', 'selftext', 'send_replies', 'spoiler', 'stickied',
               ]

    '''Columns that are too sparse (very less entries)'''

    drop_list2 = ['thumbnail', 'thumbnail_height', 'thumbnail_width', 'post_hint', 'preview', 'author_cakeday', 'retrieved_on', 'author_flair_text_color',
             'suggested_sort', 'author_flair_type'] 

    '''Columns that are redundant or correlated with other columns'''

    drop_list3 = ['parent_whitelist_status', 'subreddit_id', 'subreddit_name_prefixed', 'permalink', 'id']

    # ignore code above 3 drop lists
    master_list_to_drop = drop_list1 + drop_list2 + drop_list3

    master_list_to_drop = list(set(master_list_to_drop))

    def col_preprocess(df):
      df = df.drop(*master_list_to_drop)
      df = df.fillna( {'whitelist_status':'no_status'} )
      return df

    df_train = df_train.drop(*master_list_to_drop)
    df_test = df_test.drop(*master_list_to_drop)

    # Imputing blank values as "no_status: in whitelist_status column in both train and test dataset

    df_train = df_train.fillna( {'whitelist_status':'no_status'} )
    df_test = df_test.fillna( {'whitelist_status':'no_status'} )

    # Checking to see if any null values remain in train dataset
    df_train.select([count(when(col(c).isNull(), c)).alias(c) for c in df_train.columns])
    # Checking to see if any null values remain in test dataset
    df_test.select([count(when(col(c).isNull(), c)).alias(c) for c in df_test.columns])
    # Checking distinct values in each column of training set
    df_train.agg(*(countDistinct(col(c)).alias(c) for c in df_train.columns))

    print("No. of columns in training data set are: ", len(df_train.columns))
    print("No. of columns in test data set are: ", len(df_test.columns))	
    # Looking at the count of posts from each subreddit

    df_train.groupby('subreddit').count()  

    return df_train, df_test 

def translation(x):
  translator = Translator()
  return translator.translate(str(x), dest = 'en').text

def entropy(string):
  "Calculates the Shannon entropy of a string"
  string = string.strip()
  # get probability of chars in string
  prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
  # calculate the entropy
  entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
  return entropy

def numDigits(string):
  digits = [i for i in string if i.isdigit()]
  return len(digits)

def feature_transform(df):
    """Transform clean train or test dataset (output of clean_data()) into features"""
    ## Feature Engineering & Transformations in Train Dataset
 
    trans = udf(translation)
    spark.udf.register('trans', trans)
    entro = udf(entropy)
    spark.udf.register('entro', entro)
    digit = udf(numDigits)
    spark.udf.register('digit', digit)

    # Translating Title Column
    df_train_trans = df.withColumn('title_translated', trans('title'))

    # Performing Sentiment Analysis on Translated Title
    sent = udf(lambda x: TextBlob(x).sentiment[0])
    spark.udf.register('sentiment', sent)
    df_train_trans = df_train_trans.withColumn('title_sentiment',sent('title_translated').cast('double'))

    # Computing average score over each author and domain
    df_train_trans = df_train_trans.withColumn('avg_author_score', F.avg('score').over(Window.partitionBy('author')))
    df_train_trans = df_train_trans.withColumn('domain_avg_score', F.avg('score').over(Window.partitionBy('domain')))

    # Tokenizing translated title
    regex_tokenizer = RegexTokenizer(inputCol="title_translated", outputCol="title_tokenized", pattern="\\W")
    df_train_trans = regex_tokenizer.transform(df_train_trans)

    # Removing stopwords from tokenized title (i.e. I, or, and, etc.)
    remover = StopWordsRemover(inputCol="title_tokenized", outputCol="title_stopwords_removed")
    df_train_trans = remover.transform(df_train_trans)

    # Generating word-count, length, avg word length features on title
    df_train_trans = df_train_trans.withColumn('wordCount', F.size(F.split(F.col('title_translated'), ' ')))
    df_train_trans = df_train_trans.withColumn("length_of_title", F.length("title_translated"))
    df_train_trans = df_train_trans.withColumn('avg_word_length', (F.col('length_of_title')-F.col('wordCount')+1)/F.col('wordCount'))

    # TF-IDF on Title (stopwords removed)
    # map title to term frequencies
    hashingTF = HashingTF(inputCol="title_stopwords_removed", outputCol="title_tf", numFeatures=15)
    featurizedData = hashingTF.transform(df_train_trans)
    # alternatively, CountVectorizer can also be used to get term frequency vectors
    
    # inverse document frequency- log of documents/ document frequency
    idf = IDF(inputCol="title_tf", outputCol="title_tf_idf")
    idfModel = idf.fit(featurizedData)
    df_train_trans = idfModel.transform(featurizedData)

    # Word2Vec generation on Title (stopwords removed)
    word2Vec = Word2Vec(vectorSize=5, minCount=1, inputCol="title_stopwords_removed", outputCol="title_word2vec")
    model_w2v = word2Vec.fit(df_train_trans)
    df_train_trans = model_w2v.transform(df_train_trans)

    # Feature Engineering on "created_utc" column
    df_train_trans = df_train_trans.withColumn("date",F.to_timestamp(df_train_trans["created_utc"]))
    df_train_trans = df_train_trans.withColumn("day_of_week", date_format(col('date'), 'EEEE'))
    df_train_trans = df_train_trans.withColumn("hour", hour(col('date')))
    df_train_trans = df_train_trans.withColumn("week_day", dayofweek(df_train_trans.date))

    # Feature Engineering on "url" column
    df_train_trans = df_train_trans.withColumn('entropy', entro('url').cast('float'))
    df_train_trans = df_train_trans.withColumn('no_of_digits', digit('url').cast('float'))
    df_train_trans = df_train_trans.withColumn('url_length', F.length('url').cast('int'))
    
    df_train_trans.groupby('hour').avg('score')
    df_train_trans.groupby('day_of_week').avg('score').show()

if __name__ == '__main__':
    df_train, df_test = clean_data(spark=spark)
    df_train_trans = feature_transform(df_train)
    df_test_trans = feature_transform(df_test)
    
    df_train_trans_pd = df_train_trans.toPandas()
    plt.figure(figsize = (15,15))
    sns.heatmap(df_train_trans_pd.corr(), annot = True, fmt = '.2g')
    plt.savefig('./figures/features_cross_correlation')
 
