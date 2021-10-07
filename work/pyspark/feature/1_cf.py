#!/usr/bin/env python
# coding: utf-8
# 协同过滤-CF

# 概念
# 1、交替最小二乘法
# https://blog.csdn.net/qq_30031221/article/details/107832261
# 2、ALS在Spark上的优化
# https://blog.csdn.net/butterluo/article/details/48271361
# 3、交叉验证
# Spark ML Tuning：模型选择和超参调优
# https://www.jianshu.com/p/7e1011b135a1


# https://www.cnblogs.com/asker009/p/12426964.html

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import os
from feature import common

top_n = 10
reg_param = 0.01
max_iter = 5


if __name__ == '__main__':
    # 1、spark初始化
    spark = common.spark_init()

    # 2、特征工程
    rating_file = os.path.join(common.data_dir, 'ratings.csv')
    # 读取csv
    rating_df0 = spark.read.format('csv').option('header', 'true').load(rating_file)
    # 保证数据格式(userId int, movieId int, rating float)
    cast_sql = 'select cast(userId as int) userId,cast(movieId as int) movieId,cast(rating as float) rating from temp'
    rating_df = common.select(spark, rating_df0, cast_sql)
    rating_df.show()
    rating_df.createOrReplaceTempView
    # 训练、测试拆分
    train_df, test_df = rating_df.randomSplit((0.8, 0.2))

    # 3、推荐模型
    # 3-1、模型选择(交替最小二乘法)
    als = ALS(regParam=reg_param, maxIter=max_iter, userCol='userId', itemCol='movieId',
              ratingCol='rating', coldStartStrategy='drop')
    # 3-2、模型训练
    model = als.fit(train_df)
    # # Evaluate the model by computing the RMSE on the test data
    # 3-3、模型预测
    predicts = model.transform(test_df)
    predicts.show()
    # 3-4、评估指标(均方根误差)
    evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='rating', metricName='rmse')
    rmse = evaluator.evaluate(predicts)
    print('Root-mean-square error = {}'.format(rmse))

    # 4、模型使用
    # 模型效果数据top 10
    model.itemFactors.show(10, truncate=False)
    model.userFactors.show(10, truncate=False)
    # 电影推荐top 10
    userRecs = model.recommendForAllUsers(top_n)
    userRecs.show(5, False)
    # 用户推荐top 10
    movieRecs = model.recommendForAllItems(top_n)
    movieRecs.show(5, False)
    # 3个用户推荐
    user_df = rating_df.select(als.getUserCol()).distinct().limit(3)
    userSubsetRecs = model.recommendForUserSubset(user_df, top_n)
    userSubsetRecs.show(5, False)
    # 3部电影推荐
    movie_df = rating_df.select(als.getItemCol()).distinct().limit(3)
    movieSubSetRecs = model.recommendForItemSubset(movie_df, top_n)
    movieSubSetRecs.show(5, False)

    # 5、交叉验证
    param_grid = ParamGridBuilder().addGrid(als.regParam, [reg_param]).build()
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=10)
    cvModel = cv.fit(test_df)
    avgMetrics = cvModel.avgMetrics
    params = cvModel.getEstimatorParamMaps()

    all_params = list(zip(params, avgMetrics))
    best_param = sorted(all_params, key=lambda x: x[1], reverse=True)[0]
    for p, v in best_param.items():
        print("{} : {}".format(p.name, v))

    # 6、spark关闭
    spark.stop()
