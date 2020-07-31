
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

""" Step 4: Data exploration"""

train_set1.select([count(when(col(c).isNull(), c)).alias(c) for c in train_set1.columns]).show()
test_set1.select([count(when(col(c).isNull(), c)).alias(c) for c in test_set1.columns]).show()
train_set1.show()










#######################Code that Simon has worked on, organize later...?#################


from scipy.stats import t

def checktstat(df,colname):
  ###Function takes a data frame, and column name, and performs a t-test to assess impact on score
  ###Function returns a message, and the t-statistic
  #check number of categories
  num_cats = df.select(F.countDistinct(colname).alias("count")).collect()[0]["count"]
  

  #Check if column is categorical
  if (num_cats > 2)| (num_cats==1):
    check = "column not binary"
    useful = "NA"
  else:
    #Calculate mean by group
    mean_values = df.groupBy(colname).agg(F.mean('score').alias("mean")).collect()
    label1 = mean_values[0][colname]
    label2 = mean_values[1][colname]
    mean1 = mean_values[0]["mean"]
    mean2 = mean_values[1]["mean"]

    #Calculate standard deviation by group
    stddev_values = df.groupBy(colname).agg(F.stddev('score').alias("stddev")).collect()
    stddev1 = stddev_values[0]["stddev"]
    stddev2 = stddev_values[1]["stddev"]

    #Count group size
    counts = df.groupBy(colname).agg(F.count('score').alias("count")).collect()
    count1 = counts[0]["count"]
    count2 = counts[1]["count"]

    #calculate standard errors
    std_err1 = stddev1/(count1**0.5)
    std_err2 = stddev2/(count2**0.5)

    std_err_dif = (std_err1**2 + std_err2**2)**0.5

    #calculate t statistic
    t_stat  = (mean1 - mean2) / std_err_dif

    degrees_freedom = count1+count2-2
    alpha = 0.05
    critical = t.ppf(1.0 - alpha, degrees_freedom)

    # calculate the p-value
    p = (1 - t.cdf(abs(t_stat), degrees_freedom)) * 2
    
    #check signficance
    if abs(t_stat) <= critical:
      check = "Variable not significant"
      useful = abs(t_stat)
    else:
      check = "Variable significant"
      useful = abs(t_stat)
   
    
  return check,useful



#Uses window function to determine if domain is common (over pop_threshold)
df_train = df_train.withColumn('domaincount', F.count('domain').over(Window.partitionBy('domain')))
pop_thresh = 20
df_train = df_train.withColumn("commondomain",F.when((df_train["domaincount"]>pop_thresh),1).otherwise(0))


#One-hot encoding for subreddit on training dataset
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
stringIndexer = StringIndexer(inputCol="subreddit", outputCol="subredditIndex")
model = stringIndexer.fit(df_train)
df_train = model.transform(df_train)

num_reddits = df_train.select(F.countDistinct("subreddit")).collect()[0][0]

for i in range(num_reddits):
  col_name = "subreddit_"
  col_name = col_name + str(i)
  df_train = df_train.withColumn(col_name,F.when((df_train["subredditIndex"]==i),1).otherwise(0))


#Average domain and author score for training set
df_train = df_train.withColumn('avg_author_score', F.avg('score').over(Window.partitionBy('author')))
df_train = df_train.withColumn('domain_avg_score', F.avg('score').over(Window.partitionBy('domain')))
  
  
  

#Uses training data set and joins together with test set to assign known author and domain average scores
#drops duplicate columns, and imputes average values to eliminate nulls values

from pyspark.sql.functions import col

# df_joined = df_test.alias('a').join(df_train.alias('b'),col('b.author') == col('a.author')).select([col('a.'+xx) for xx in a.columns] + [col('b.avg_author_score')])
df_author_scores = df_train['author','avg_author_score'].distinct()
df_domain_scores = df_train['domain','domain_avg_score'].distinct()

domain_avg_avg = df_train.select(F.avg(df_train["domain_avg_score"])).collect()[0][0]
author_avg_avg = df_train.select(F.avg(df_train["avg_author_score"])).collect()[0][0]

df_author_scores = df_author_scores.selectExpr("author as authorlookup", "avg_author_score as avg_author_score")
df_domain_scores = df_domain_scores.selectExpr("domain as domainlookup", "domain_avg_score as domain_avg_score")

#df_joined2 = df_test.join(df_train, df_test.author == df_train.author,how='inner').select(df_test["*"],df_train["avg_author_score"])
df_test = df_test.join(df_author_scores, df_test.author == df_author_scores.authorlookup,how='left')
df_test = df_test.join(df_domain_scores, df_test.domain == df_domain_scores.domainlookup,how='left')

df_test = df_test.drop("authorlookup")
df_test = df_test.drop("domainlookup")

#Impute average values over null values
df_test = df_test.fillna({'domain_avg_score':domain_avg_avg})
df_test = df_test.fillna({'avg_author_score':author_avg_avg})



#Join common domain and test data, impute 0 if null (not common)
df_common_domain = df_train['domain','commondomain'].distinct()
df_common_domain = df_common_domain.selectExpr("domain as domainlookup", "commondomain as commondomain")

df_test = df_test.join(df_common_domain, df_test.domain == df_common_domain.domainlookup,how='left')
df_test = df_test.drop("domainlookup")
df_test = df_test.fillna({'commondomain':0})

#need to one-hot encode test set subreddits
