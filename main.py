from pyspark.ml.feature import CountVectorizer, StringIndexer, RegexTokenizer,StopWordsRemover, OneHotEncoder, HashingTF, IDF, Word2Vec, VectorAssembler, PCA
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import numpy as np

def random_forest_regressor(result_trainpca, result_testpca):
    rf = RandomForestRegressor(featuresCol='pcaFeatures') # featuresCol="indexedFeatures",numTrees=2, maxDepth=2, seed=42

    df_train_sel = result_trainpca.select(col("pcaFeatures"), col("score").alias("label"))
    # df_train_sel.show(3)
    df_test_sel = result_testpca.select(col("pcaFeatures"), col("score").alias("label"))
    # df_test_sel.show(3)

    model = rf.fit(df_train_sel)
    predictions = model.transform(df_test_sel)

    # Select example rows to display.
    predictions.select("pcaFeatures","label", "prediction").show(5)

    rf_evaluator = RegressionEvaluator(predictionCol="prediction", \
                     labelCol="label",metricName="r2")
    print("R Squared (R2) on test data = %g" % rf_evaluator.evaluate(predictions))

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    # grid search hyperparameter tuning
    paramGrid= ParamGridBuilder()\
        .addGrid(rf.numTrees, [2, 4, 8, 16])\
        .addGrid(rf.maxDepth, [2, 4, 8, 16])\
        .build()
    # cross validation hyperparameter selection
    crossval = CrossValidator(
                    estimator=rf,
                    estimatorParamMaps=paramGrid,
                    evaluator=rf_evaluator,
                    numFolds=10)
    cvModel = crossval.fit(df_train_sel)
    predictions_cv = cvModel.transform(df_test_sel)
    print("R2 on test data (cross-validated) = %g" % rf_evaluator.evaluate(predictions_cv))
    print(cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)])
    return cvModel 

def linear_regressor(result_trainpca, result_testpca):
    lr = LinearRegression(featuresCol = 'pcaFeatures', labelCol='score', maxIter=1000, regParam=0.3, elasticNetParam=0.8)
    
    lr_model = lr.fit(result_trainpca)
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))

    trainingSummary = lr_model.summary
    # On Train Set
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)

    lr_predictions = lr_model.transform(result_testpca)
    lr_predictions.select("prediction","score","pcaFeatures").show(5)

    lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                     labelCol="score",metricName="r2")

    # On Test Set
    print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

    test_result = lr_model.evaluate(result_testpca)
    print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)
    
    # grid search hyperparameter tuning
    paramGrid= ParamGridBuilder()\
        .addGrid(lr.regParam, [0.3, 0.5, 1.0, 2.0, 4.0])\
        .addGrid(lr.elasticNetParam, [0.1, 0.2, 0.4, 0.8])\
        .build()
    # cross validation hyperparameter selection
    crossval = CrossValidator(
                    estimator=lr,
                    estimatorParamMaps=paramGrid,
                    evaluator=lr_evaluator,
                    numFolds=10)
    cvModel = crossval.fit(result_trainpca)
    predictions_cv = cvModel.transform(result_testpca)
    print("R2 on test data (cross-validated) = %g" % lr_evaluator.evaluate(predictions_cv))
    print(cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)])
    return cvModel 

def decision_tree_regressor(result_trainpca, result_testpca):
    dt = DecisionTreeRegressor(featuresCol="pcaFeatures")

    df_train_sel = result_trainpca.select(col("pcaFeatures"), col("score").alias("label"))
    # df_train_sel.show(3)
    df_test_sel = result_testpca.select(col("pcaFeatures"), col("score").alias("label"))
    # df_test_sel.show(3)

    model_dt = dt.fit(df_train_sel)
    predictions_dt = model_dt.transform(df_test_sel)

    predictions_dt.select("pcaFeatures","label","prediction").show(5)

    rf_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="r2")
    print("R Squared (R2) on test data = %g" % rf_evaluator.evaluate(predictions_dt))

    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse_dt = evaluator.evaluate(predictions_dt)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse_dt)

    # grid search hyperparameter tuning
    paramGrid= ParamGridBuilder()\
        .addGrid(dt.maxBins, [16, 32, 64, 128])\
        .addGrid(dt.maxDepth, [2, 4, 8, 16])\
        .build()
    # cross validation hyperparameter selection
    crossval = CrossValidator(
                    estimator=dt,
                    estimatorParamMaps=paramGrid,
                    evaluator=rf_evaluator,
                    numFolds=10)
    cvModel = crossval.fit(df_train_sel)
    predictions_cv = cvModel.transform(df_test_sel)
    print("R2 on test data (cross-validated) = %g" % rf_evaluator.evaluate(predictions_cv))
    print(cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)])
    return cvModel 
if __name__ == '__main__':
    df_train_trans = spark.read.json("train.json")
    df_test_trans = spark.read.json("test.json")

    # Initializing assembler to tie-up all relevant features together
    vectorAssembler = VectorAssembler(inputCols = ["all_ads", "archived", "avg_word_length", "brand_safe", "can_gild", "entropy", "hour", \
                                                  "is_crosspostable", "length_of_title", "no_follow", "no_of_digits", "num_comments",\
                                                  "over_18", "promo_adult_nsfw", "public", "subreddit_0", "subreddit_1", \
                                                  "subreddit_2", "subreddit_3","subreddit_4", "subreddit_5","subreddit_6", "subreddit_7",\
                                                  "subreddit_8", "subreddit_9","subreddit_10", "subreddit_11","subreddit_12", "subreddit_13",\
                                                  "subreddit_14", "subreddit_15","subreddit_16", "subreddit_17","subreddit_18", "subreddit_19",\
                                                  "subreddit_20", "subreddit_21","subreddit_22", "title_ner_counts", "title_sentiment",\
    #                                               "title_tf_idf", "title_word2vec", \
                                                   "url_length", "week_day", "wordCount"]\
                                      , outputCol = 'features')

    # Assembling all stated features above on train and test set
    df_train_assem = vectorAssembler.transform(df_train_trans)
    df_test_assem = vectorAssembler.transform(df_test_trans)

    # Run PCA on all features
    pca = PCA(k=6, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(df_train_assem)
    result_trainpca = model.transform(df_train_assem)
    result_testpca = model.transform(df_test_assem)
    result_testpca = result_testpca.select("pcaFeatures", "score")
    result_trainpca = result_trainpca.select("pcaFeatures", "score")

    # rf_model = random_forest_regressor(result_trainpca, result_testpca)
    # lr_model = linear_regressor(result_trainpca, result_testpca)
    dt_model = decision_tree_regressor(result_trainpca, result_testpca)
