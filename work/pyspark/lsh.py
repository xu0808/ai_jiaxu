#!/usr/bin/env python
# coding: utf-8
# 局部敏感哈希
# https://www.jianshu.com/p/4729e83b7b48

import datetime
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.sql import SparkSession


def now():
    return datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')


if __name__ == '__main__':
    print('%s spark init!' % now())
    # 1、spark初始化
    spark = SparkSession.builder \
        .appName('rec_spark') \
        .master("local[2]") \
        .config("spark.v0.maxToStringFields", "100") \
        .config("spark.default.parallelism", "2") \
        .config("spark.sql.shuffle.partitions", "2") \
        .enableHiveSupport() \
        .getOrCreate()

    # 2、初始化数据
    dataA = [(0, Vectors.dense([1.0, 1.0]),),
             (1, Vectors.dense([1.0, -1.0]),),
             (2, Vectors.dense([-1.0, -1.0]),),
             (3, Vectors.dense([-1.0, 1.0]),)]
    dfA = spark.createDataFrame(dataA, ["id", "features"])
    dataB = [(4, Vectors.dense([1.0, 0.0]),),
             (5, Vectors.dense([-1.0, 0.0]),),
             (6, Vectors.dense([0.0, 1.0]),),
             (7, Vectors.dense([0.0, -1.0]),)]
    dfB = spark.createDataFrame(dataB, ["id", "features"])

    # 输入表示欧几里得空间一个点
    # bucketLength：更大的桶降低了假阴性率
    # numHashTables：默认为1.哈希表的数量，增大可以提高准确率但增加运行时间
    # 另外，还有基于Jaccard距离的MinHashLSH，将每个数据集表示为一个二进制稀疏向量
    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=2.0,
                                      numHashTables=3)
    model = brp.fit(dfA)
    # Feature Transformation
    print("The hashed dataset where hashed values are stored in the column 'hashes':")
    model.transform(dfA).show()

    # Compute the locality sensitive hashes for the input rows, then perform approximate similarity join.
    # We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    # `model.approxSimilarityJoin(transformedA, transformedB, 1.5)`
    print("Approximately joining dfA and dfB on Euclidean distance smaller than 1.5:")  # 计算欧式距离（阈值小于1.5才纳入）
    # 2个数据集
    model.approxSimilarityJoin(dfA, dfB, 1.5, distCol="EuclideanDistance") \
        .select(col("datasetA.id").alias("idA"),
                col("datasetB.id").alias("idB"),
                col("EuclideanDistance")).show()

    # Compute the locality sensitive hashes for the input rows, then perform approximate nearest neighbor search.
    # We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    # `model.approxNearestNeighbors(transformedA, key, 2)`
    key = Vectors.dense([1.0, 0.0])
    print("Approximately searching dfA for 2 nearest neighbors of the key:")
    # 从dfA中找到2个与key距离最近的点
    model.approxNearestNeighbors(dfA, key, 2).show()

    spark.stop()
