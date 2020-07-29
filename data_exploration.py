
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

import pandas as pd

# sc = SparkContext()
# spark = SparkSession(sc)
# sqlc = SQLContext(sc)
# print(sc.getConf().getAll())     

"""STEP 2: importing data"""

df_train = spark.read.format('json').option('inferSchema', 'false').option('header', 'false').option('sep', ',').load('RS_v2_2006-03')
df_test = spark.read.format('json').option('inferSchema', 'false').option('header', 'false').option('sep', ',').load('RS_v2_2006-04')

"""STEP 3: CLEANING DATA"""

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
master_list_to_drop = list(set(master_list_to_drop))

def col_preprocess(df):
  df = df.drop(*master_list_to_drop)
  df = df.fillna( {'whitelist_status':'no_status'} )
  return df

train_set1 = col_preprocess(df_train)
test_set1 = col_preprocess(df_test)


print("No. of columns in training data set are: ", len(train_set1.columns))
print("No. of columns in test data set are: ", len(test_set1.columns))

""" STEP 4: Data exploration"""

train_set1.select([count(when(col(c).isNull(), c)).alias(c) for c in train_set1.columns]).show()
test_set1.select([count(when(col(c).isNull(), c)).alias(c) for c in test_set1.columns]).show()
train_set1.show()

train_set_pd = train_set1.toPandas()
""" Correlation matrix of numerical/boolean data"""
train_set_pd.corr() 

import matplotlib.pyplot as plt
plt.figure(figsize=[30,20])
train_set_pd[['author', 'score']].sort_values(by='author')[0:1000].boxplot(by='author') 
plt.xticks(rotation=90) 
plt.tight_layout()
plt.savefig('boxplot_score_groupedby_author')
