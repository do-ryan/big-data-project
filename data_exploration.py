
from pyspark.sql import Row
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import FloatType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import LongType
from pyspark.sql.functions import *
from pyspark.sql import Window
from pyspark.mllib.stat import Statistics

import pandas as pd

from preprocess import clean_data

# sc = SparkContext()
# spark = SparkSession(sc)
# sqlc = SQLContext(sc)
# print(sc.getConf().getAll())

df_train, df_test = clean_data(spark=spark)     

""""""""""""""""""""
""" STEP 4: Data exploration"""
""""""""""""""""""""
# Plotting correlation matrix

features = df_train.select(["brand_safe", "can_gild", "is_crosspostable", "no_follow", "num_comments", "over_18", "score"]).rdd.map(lambda row: row[0:])
corr_mat=Statistics.corr(features, method="pearson")
sc.parallelize(corr_mat).map(lambda x: x.tolist()).toDF(["brand_safe", "can_gild", "is_crosspostable", "no_follow", "num_comments", "over_18", "score"]).show()

train_set_pd = df_train.toPandas()
""" Correlation matrix of numerical/boolean data"""
train_set_pd.corr() 

df_train_trans = spark.read.json('train.json')
df_test_trans = spark.read.json('test.json')
df_train_trans_pd = df_train_trans.toPandas()

""" Scores grouped by author/ domain"""
import matplotlib.pyplot as plt
plt.figure(figsize=[30,40])
train_set_pd[['author', 'score']].sort_values(by='author')[0:300].boxplot(by='author', showfliers=False) 
plt.xticks(rotation=90) 
plt.gcf().subplots_adjust(bottom=0.3)
plt.savefig('./figures/boxplot_score_groupedby_author')

plt.figure(figsize=[40, 80]) 
train_set_pd[['domain', 'score']].sort_values(by='domain')[0:150].boxplot(by='domain', showfliers=False) 
plt.gcf().subplots_adjust(bottom=0.4) 
plt.xticks(rotation=90) 
plt.xlabel('domains', fontsize=5)
plt.savefig('./figures/boxplot_score_groupedby_domain') 

plt.figure(figsize=[40, 40]) 
train_set_pd[['subreddit', 'score']].sort_values(by='subreddit')[0:7000].boxplot(by='subreddit', showfliers=False) 
plt.gcf().subplots_adjust(bottom=0.4) 
plt.xlabel('subreddit', fontsize=9)
plt.xticks(rotation=90) 
plt.savefig('./figures/boxplot_score_groupedby_subreddit') 

plt.figure(figsize=[40, 40]) 
train_set_pd[['subreddit_type', 'score']].sort_values(by='subreddit_type').boxplot(by='subreddit_type', showfliers=False) 
plt.gcf().subplots_adjust(bottom=0.4) 
plt.xticks(rotation=90) 
plt.savefig('./figures/boxplot_score_groupedby_subreddit_type') 

plt.figure(figsize=[40, 40]) 
df_train_trans_pd[['day_of_week', 'score']].boxplot(by='day_of_week', showfliers=False) 
plt.gcf().subplots_adjust(bottom=0.4) 
plt.xticks(rotation=90) 
plt.savefig('./figures/boxplot_score_groupedby_dayofweek') 

plt.figure(figsize=[40, 40]) 
df_train_trans_pd[['hour', 'score']].boxplot(by='hour', showfliers=False) 
plt.gcf().subplots_adjust(bottom=0.4) 
plt.xticks(rotation=90) 
plt.xlabel('hour', fontsize=8)
plt.savefig('./figures/boxplot_score_groupedby_hour') 

""" Score column (y label) histogram """
plt.figure()
gre_histogram = df_train.select('score').rdd.flatMap(lambda x: x).histogram(sc.parallelize(range(0, 601, 25)).collect())

# Loading the Computed Histogram into a Pandas Dataframe for plotting
pd.DataFrame(list(zip(*gre_histogram)), columns=['bin', 'frequency']).set_index('bin').plot(kind='bar')
plt.savefig('./figures/score_histogram')

""" Scatter score vs utf """
plt.figure()  
train_set_pd[['created_utc', 'score']].plot(x='created_utc', y='score', kind='scatter')
plt.xticks(rotation=90)  
plt.gcf().subplots_adjust(bottom=0.3)  
plt.savefig('./figures/scatter_createdutf_score')  

""" Scatter score vs num comments """
plt.figure(figsize=[30, 30])  
train_set_pd[['num_comments', 'score']].plot(x='num_comments', y='score', kind='scatter') 
plt.xticks(rotation=90)  
plt.gcf().subplots_adjust(bottom=0.3)  
plt.savefig('./figures/scatter_num_comments_score')  














