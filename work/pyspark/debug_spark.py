#!/usr/bin/env python
# coding: utf-8
# spark调试

from pyspark.sql.types import *
import datetime
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import *


def now():
    return datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')


def item_info(item, weight):
    return "{'i':%s, 'e':%f}" % (str(item), weight)


def list_str(rec_list):
    return "[%s]" % ','.join(rec_list)


def uip_dict(index, rec_list):
    return json.dumps({'code': index, 'rec_list': rec_list})


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
    print('%s rec_df  start!' % now())
    sql = "select '1234' userid,'12345' i, 0.55 w"
    sql += " union select '1234' userid,'12346' i, 0.66 w"
    sql += " union select '1234' userid,'12347' i, 0.77 w"
    sql += " union select '1234' userid,'12348' i, 0.88 w"
    rec_df_0 = spark.sql(sql).repartition(2)
    rec_df_0.show(20, False)
    rec_df_0.printSchema()
    print('%s rec_df  end!' % now())

    print('%s rec_list  start!' % now())
    rec_df_0.createOrReplaceTempView('rec')
    spark.udf.register('item_info', item_info, StringType())
    spark.udf.register('list_str', list_str, StringType())
    sql = 'select userid,list_str(collect_set(item_info(i, w))) rec_list from rec group by userid'
    rec_df_1 = spark.sql(sql).repartition(2)
    rec_df_1.show(1, False)
    rec_df_1.printSchema()
    print('%s rec_list  end!' % now())

    print('%s uip format start!' % now())
    rec_df_1.createOrReplaceTempView('rec_list')
    spark.udf.register('uip_dict', uip_dict, StringType())
    sql = 'select userid,uip_dict(\'index_code\', rec_list) rec_list from rec_list'
    rec_df_2 = spark.sql(sql).repartition(2)
    rec_df_2.show(1, False)
    rec_df_2.printSchema()
    print('%s uip format end!' % now())

    spark.stop()
