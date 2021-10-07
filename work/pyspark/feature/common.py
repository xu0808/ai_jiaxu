#!/usr/bin/env python
# coding: utf-8
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import *

is_local = True
is_aicloud = False
# 调试时进行采样小规模数据等
is_debug = False

# 数据文件目录
data_dir = 'E:\\workspace\\rec_server\\src\\main\\resources\\webroot\\sampledata'

# 本地化配置
if is_local:
    # os.environ['JAVA_HOME'] = r'D:\work\ide\Java\jdk1.8.0_261'
    # os.environ['SPARK_HOME'] = r'D:\soft\src\spark-2.4.7-bin-hadoop2.7'
    sys.path.append(r'D:\soft\src\spark-2.4.7-bin-hadoop2.7\python')
    data_dir = 'E:\\workspace\\rec_server\\src\\main\\resources\\webroot\\sampledata'

if is_debug:
    data_dir = '/data'

out_dir = os.path.join('/data', 'out')


def spark_init():
    if is_local:
        spark = SparkSession.builder \
            .appName('rec_spark') \
            .master("local[8]") \
            .config("spark.v0.maxToStringFields", "100") \
            .enableHiveSupport() \
            .getOrCreate()
    else:
        spark = SparkSession.builder \
            .appName('rec_spark') \
            .config('spark.driver.maxResuItSize', '5g') \
            .config('spark-executor.memory', '4g') \
            .enableHiveSupport() \
            .getOrCreate()
    return spark


# data基本处理-默认使用temp表名
def select(spark: SparkSession, df0: DataFrame, sql: str):
    df0.createOrReplaceTempView('temp')
    return spark.sql(sql)


# 读取本地文件
def read_csv(spark: SparkSession, file_name: str):
    return spark.read.format('csv').option('header', 'true') \
        .load(os.path.join(data_dir, file_name))


# 读取本地文件
def read_movie(spark: SparkSession):
    return read_csv(spark, 'movies.csv')


# 读取本地文件
def read_ratings(spark: SparkSession):
    return read_csv(spark, 'ratings.csv')
