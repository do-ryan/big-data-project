# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

dbutils.library.installPyPI("mlflow")
dbutils.library.restartPython()
import mlflow

# COMMAND ----------

from pyspark.sql.functions import isnan, when, count, col

from pyspark.sql import functions as F 
from pyspark.sql import Window

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/RS_v2_2006_03"
file_type = "json"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_train = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df_train)

# COMMAND ----------

len(df_train.columns)

# COMMAND ----------

columns_to_drop = ['archived', 'author_cakeday', 'author_flair_text_color', 'author_flair_background_color', 'author_flair_css_class', \
                   'author_flair_richtext', 'author_flair_text', 'author_flair_type', 'contest_mode', 'edited', 'gilded', 'hidden', 'hide_score', \
                   'is_reddit_media_domain', 'is_self', 'is_video', 'distinguished', 'link_flair_css_class', 'link_flair_richtext', \
                   'link_flair_text', 'post_hint', 'link_flair_text_color', 'link_flair_type', 'locked', 'media', 'rte_mode', 'preview', \
                   'retrieved_on', 'secure_media', 'selftext', 'send_replies', 'spoiler', 'stickied', 'thumbnail', 'thumbnail_height', \
                   'thumbnail_width', 'whitelist_status', 'num_crossposts', 'suggested_sort']

df_train = df_train.drop(*columns_to_drop)

# COMMAND ----------

display(df_train)

# COMMAND ----------

len(df_train.columns)

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/RS_v2_2006_04-1"
file_type = "json"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df_test = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df_test)

# COMMAND ----------

len(df_test.columns)

# COMMAND ----------

# df.drop('author_cakeday').collect()

columns_to_drop = ['archived', 'author_cakeday', 'author_flair_text_color', 'author_flair_background_color', 'author_flair_css_class', \
                   'author_flair_richtext', 'author_flair_text', 'author_flair_type', 'contest_mode', 'edited', 'gilded', 'hidden', 'hide_score', \
                   'is_reddit_media_domain', 'is_self', 'is_video', 'distinguished', 'link_flair_css_class', 'link_flair_richtext', \
                   'link_flair_text', 'post_hint', 'link_flair_text_color', 'link_flair_type', 'locked', 'media', 'rte_mode', 'preview', \
                   'retrieved_on', 'secure_media', 'selftext', 'send_replies', 'spoiler', 'stickied', 'thumbnail', 'thumbnail_height', \
                   'thumbnail_width', 'whitelist_status', 'num_crossposts', 'suggested_sort']

df_test = df_test.drop(*columns_to_drop)

# COMMAND ----------

display(df_test)

# COMMAND ----------

len(df_test.columns)

# COMMAND ----------

df_train = df_train.fillna( {'parent_whitelist_status':'no_status'} )
df_test = df_test.fillna( {'parent_whitelist_status':'no_status'} )

# COMMAND ----------

display(df_train.select([count(when(col(c).isNull(), c)).alias(c) for c in df_train.columns]))

# COMMAND ----------

display(df_test.select([count(when(col(c).isNull(), c)).alias(c) for c in df_test.columns]))

# COMMAND ----------

columns_to_drop2 = ['created_utc', 'id', 'permalink', 'subreddit_id', 'subreddit_name_prefixed', 'url']

df_train = df_train.drop(*columns_to_drop2)

# COMMAND ----------

len(df_train.columns)

# COMMAND ----------

columns_to_drop2 = ['created_utc', 'id', 'permalink', 'subreddit_id', 'subreddit_name_prefixed', 'url']

df_test = df_test.drop(*columns_to_drop2)

# COMMAND ----------

len(df_test.columns)

# COMMAND ----------

display(df_train)

# COMMAND ----------

df_train_p  = df_train.toPandas()

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer

regexTokenizer = RegexTokenizer(gaps = False, pattern = '\w+', inputCol = 'domain', outputCol = 'domain_tok')

df_train = regexTokenizer.transform(df_train)
display(df_train)

# COMMAND ----------

# remove stopwords
from pyspark.ml.feature import StopWordsRemover
swr = StopWordsRemover(inputCol = 'domain_tok', outputCol = 'domain_tok_sw')

df_train = swr.transform(df_train)
display(df_train)
# reviews_swr.write.saveAsTable('reviews_swr', mode = 'overwrite')

# COMMAND ----------

from pyspark.ml.feature import Word2Vec

#create an average word vector for each document (works well according to Zeyu & Shu)
word2vec = Word2Vec(vectorSize = 10, minCount = 5, inputCol = 'domain_tok_sw', outputCol = 'domain_tok_sw_w2v')
model = word2vec.fit(df_train)
df_train = model.transform(df_train)

# display(result)
df_train.show(1, truncate = True)

# COMMAND ----------

regexTokenizer2 = RegexTokenizer(gaps = False, pattern = '\w+', inputCol = 'title', outputCol = 'title_tok')

df_train = regexTokenizer2.transform(df_train)

swr2 = StopWordsRemover(inputCol = 'title_tok', outputCol = 'title_tok_sw')

df_train = swr2.transform(df_train)

word2vec2 = Word2Vec(vectorSize = 10, minCount = 5, inputCol = 'title_tok_sw', outputCol = 'title_tok_sw_w2v')
model2 = word2vec2.fit(df_train)
df_train = model2.transform(df_train)

df_train.show(1)

# COMMAND ----------

regexTokenizer_t = RegexTokenizer(gaps = False, pattern = '\w+', inputCol = 'title', outputCol = 'title_tok')

df_test = regexTokenizer_t.transform(df_test)

swr_t = StopWordsRemover(inputCol = 'title_tok', outputCol = 'title_tok_sw')

df_test = swr_t.transform(df_test)

word2vec_t = Word2Vec(vectorSize = 10, minCount = 5, inputCol = 'title_tok_sw', outputCol = 'title_tok_sw_w2v')
model_t = word2vec_t.fit(df_test)
df_test = model_t.transform(df_test)

df_test.show(1)

# COMMAND ----------

regexTokenizer_t2 = RegexTokenizer(gaps = False, pattern = '\w+', inputCol = 'domain', outputCol = 'domain_tok')

df_test = regexTokenizer_t2.transform(df_test)

swr_t2 = StopWordsRemover(inputCol = 'domain_tok', outputCol = 'domain_tok_sw')

df_test = swr_t2.transform(df_test)

word2vec_t2 = Word2Vec(vectorSize = 10, minCount = 5, inputCol = 'domain_tok_sw', outputCol = 'domain_tok_sw_w2v')
model_t2 = word2vec_t2.fit(df_test)
df_test = model_t2.transform(df_test)

df_test.show(1)

# COMMAND ----------

df_train.columns

# COMMAND ----------

columns_to_drop = ['author', 'domain', 'title', 'domain_tok', 'domain_tok_sw', 'title_tok', 'title_tok_sw']

df_train = df_train.drop(*columns_to_drop)
df_test = df_test.drop(*columns_to_drop)

# COMMAND ----------

len(df_train.columns), len(df_test.columns)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['brand_safe', 'can_gild', 'is_crosspostable', 'no_follow', 'over_18', \
                                     'domain_tok_sw_w2v', 'title_tok_sw_w2v'], outputCol = 'features')
df_t = vectorAssembler.transform(df_train)
df_t = df_t.select(['features', 'score'])
df_t.show(3)

# COMMAND ----------

vectorAssembler_t = VectorAssembler(inputCols = ['brand_safe', 'can_gild', 'is_crosspostable', 'no_follow', 'over_18', \
                                     'domain_tok_sw_w2v', 'title_tok_sw_w2v'], outputCol = 'features')
df_t2 = vectorAssembler_t.transform(df_test)
df_t2 = df_t2.select(['features', 'score'])
df_t2.show(3)

# COMMAND ----------

display(df_t)

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='score', maxIter=1000, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(df_t)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

# COMMAND ----------

trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# COMMAND ----------

lr_predictions = lr_model.transform(df_t2)
lr_predictions.select("prediction","score","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="score",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

# COMMAND ----------

test_result = lr_model.evaluate(df_t2)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)

# COMMAND ----------

print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()

# COMMAND ----------

display(df_t)

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

rf = RandomForestRegressor(labelCol="score", featuresCol="features")

rf_model = rf.fit(df_t)

predictions = rf_model.transform(df_t2)

# COMMAND ----------

display(predictions)

# COMMAND ----------

import mlflow
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[rf])

from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.tuning import ParamGridBuilder
import numpy as np

paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 10, stop = 50, num = 3)]) \
    .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 25, num = 3)]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3)

cvModel = crossval.fit(df_t)

predictions = cvModel.transform(df_t2)

# COMMAND ----------

import matplotlib.pyplot as plt

evaluator = RegressionEvaluator(labelCol="score", predictionCol="prediction", metricName="rmse")

rmse = evaluator.evaluate(predictions)

rfPred = cvModel.transform(df)

rfResult = rfPred.toPandas()

plt.plot(rfResult.label, rfResult.prediction, 'bo')
plt.xlabel('Price')
plt.ylabel('Prediction')
plt.suptitle("Model Performance RMSE: %f" % rmse)
plt.show()

# COMMAND ----------

display(df_train)

# COMMAND ----------

import seaborn as sns

sns.boxplot("score", data=df_train_p[df_train_p["score"]<=200])

# COMMAND ----------

display(df_train.select("*").where(df_train.score <= 200))

# COMMAND ----------

df_train_num = df_train.select("brand_safe", "can_gild", "is_crosspostable", "num_comments", "no_follow", "over_18",  "score")
display(df_train_num)

# COMMAND ----------

display(features.collect())

# COMMAND ----------

features = df_train_num.rdd.map(lambda row: row[0:])

from pyspark.mllib.stat import Statistics

corr_mat=Statistics.corr(features, method="pearson")

# COMMAND ----------

print(corr_mat)

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# convert to vector column first
vector_col = "score"
assembler = VectorAssembler(inputCols=df_train_num.columns, outputCol=vector_col)
df_vector = assembler.transform(df_train_num).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col)

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

var = 'score'
x = df_train[var].values
bins = np.arange(0, 100, 5.0)

plt.figure(figsize=(10,8))
# the histogram of the data
plt.hist(x, bins, alpha=0.8, histtype='bar', color='gold',
         ec='black',weights=np.zeros_like(x) + 100. / x.size)

plt.xlabel(var)
plt.ylabel('percentage')
plt.xticks(bins)
plt.show()

# COMMAND ----------

df_train["score"].values

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `RS_v2_2006_03`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "RS_v2_2006_03"

# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------


