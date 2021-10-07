#!/usr/bin/env python
# coding: utf-8
# 特征工程

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, \
    StringIndexer, QuantileDiscretizer, MinMaxScaler
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark.sql.functions import *
import os
from feature import common

# 数据文件
ratings_file = os.path.join(common.data_dir, 'ratings.csv')
movie_file = os.path.join(common.data_dir, 'movies.csv')


def array_2_vec(genre_indexes, index_size):
    genre_indexes.sort()
    fill_list = [1.0 for _ in range(len(genre_indexes))]
    return Vectors.sparse(index_size, genre_indexes, fill_list)


def title(line):
    # 默认值1990
    if not line or len(line.strip()) < 6:
        return line
    else:  # 后5位为年度
        return line.strip()[:-6]


def year(line):
    # 默认值1990
    if not line or len(line.strip()) < 6:
        return 1990
    else:  # 后5位为年度
        return int(line.strip()[-5:-1])


def genre(genres, index):
    if not genres or len(genres.strip()) == 0:
        return None
    array = genres.split('|')
    return array[index] if len(array) > index else None


def genre_sort(genres_list):
    """
    :param
    ["Action|Adventure|Sci-Fi|Thriller", "Crime|Horror|Thriller"]
     ==>
    (('Thriller',2),('Action',1),('Sci-Fi',1),('Horror', 1), ('Adventure',1),('Crime',1))
    :return:
    ['Thriller','Action','Sci-Fi','Horror','Adventure','Crime']
    """
    genres_dict = {}
    for genres in genres_list:
        # null 直接跳过
        if not genres or len(genres.strip()) == 0:
            continue
        for g in genres.split('|'):
            if g not in genres_dict:
                genres_dict[g] = 0
            genres_dict[g] += 1
    # 按照value排序
    sorted_genres = sorted(genres_dict.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_genres]


def one_hot(movies):
    """
    独热编码
    :param movies:
    :return:
    """
    encoder = OneHotEncoderEstimator(inputCols=['movieId'], outputCols=['movieIdVector'], dropLast=False)
    # 训练模型
    encoder_model = encoder.fit(movies)
    # 使用模型
    encoder_df = encoder_model.transform(movies)
    encoder_df.printSchema()
    encoder_df.show(10)


def multi_hot(movies):
    """
    多热编码
    :param movies:
    :return:
    """
    # genreInde非整型
    genre_indexer = StringIndexer(inputCol='genre', outputCol='genre_index')
    # 训练模型
    indexer_model = genre_indexer.fit(movies)
    # 使用模型
    genre_index = indexer_model.transform(movies)
    genre_index.printSchema()
    genre_index.show(10)

    genres_index_sql = 'select movieId,collect_list(cast(genre_index as int)) genre_indexs from temp group by movieId'
    genres_index = common.select(spark, genre_index, genres_index_sql)
    genres_index.printSchema()
    genres_index.show(10)

    index_size = int(genre_index.agg(max(f.col('genre_index'))).head()[0]) + 1
    genres_vector_sql = 'select movieId,genre_indexs,array_2_vec(genre_indexs,%d) vector from temp' % index_size
    genres_vector = common.select(spark, genres_index, genres_vector_sql)
    genres_vector.show(10)


def numerical(ratings):
    """
    数值型特征的处理 - 归一化和分桶
    :param ratings: 评分数据集
    :return:
    """
    # 计算均值和组间方差
    agg_sql = 'select movieId,count(rating) num,avg(rating) avg,variance(rating) var from temp group by movieId'
    ratings_df = common.select(spark, ratings, agg_sql)
    ratings_df.show()

    movie_df = common.select(spark, ratings_df, 'select movieId,num,avg,var,vector(avg) avgVec from temp')
    movie_df.show()

    # bucketing
    discretizer = QuantileDiscretizer(numBuckets=100, inputCol='num', outputCol='bucket')
    # Normalization
    scaler = MinMaxScaler(inputCol='avgVec', outputCol='scaleAvg')
    # 训练模型
    pipeline_model = Pipeline(stages=[discretizer, scaler]).fit(movie_df)
    # 使用模型
    feature_df = pipeline_model.transform(movie_df)
    feature_df.printSchema()
    feature_df.show(10)


def label(ratings):
    """
    评分大于3.5表示喜欢
    :param ratings: 评分数据集
    :return:
    """
    # 查看评分分布情况
    total = ratings.count()
    print('total = %d' % total)
    percent_sql = 'select rating,format_number(count(*)/%d,4) percent from temp group by rating' % total
    percent_df = common.select(spark, ratings, percent_sql)
    percent_df.show()

    # 评分大于3.5表示喜欢
    label_sql = 'select userId,movieId,rating,timestamp,case when rating>=3.5 then 1 else 0 end as label from temp'
    label_df = common.select(spark, ratings, label_sql)
    label_df.show()
    return label_df


def movie_features(movies, ratings):
    """
    电影特征
    1、电影id、标题、年度、风格1、风格2、风格3
    2、电影评分记录数、评分平均值、评分标准差
    """
    # 电影id、标题、年度、风格1、风格2、风格3
    movie_sql = 'select movieId,title(title) title,year(title) year,genres,genre(genres,0) movieGenre1,' \
                'genre(genres,1) movieGenre2,genre(genres,2) movieGenre3 from temp'
    movie_df = common.select(spark, movies, movie_sql)
    movie_df.show()

    # 电影评分记录数、评分平均值、评分标准差
    rating_sql = 'select movieId,count(rating) movieRatingCount,' \
                 'format_number(avg(rating),4) movieRatingAvg,' \
                 'format_number(stddev(rating),4) movieRatingStddev ' \
                 'from temp group by movieId'
    rating_df = common.select(spark, ratings, rating_sql)
    rating_df.show()

    feature_df = ratings.join(movie_df, on=['movieId'], how='left').join(rating_df, on=['movieId'], how='left')\
        .withColumn('label', when(f.col('rating') >= 3.5, 1).otherwise(0))
    feature_df.show()
    return feature_df


def user_features(features):
    """
    用户特征
    :param movies:
    :param rating_label:
    :return:
    """
    # 1、用户最新5个评分电影
    # 2、用户近100条记录风格偏好排序
    # 3、用户近100条记录聚合查询（评分次数、电影年份均值、电影年份标准差、评分均值、评分标准差）
    over_sql = 'over(partition by userId order by timestamp rows between %d preceding and current row)'
    over_row5, over_row100 = over_sql % 5, over_sql % 100
    user_feature_sql = 'select *,collect_list(case when label=1 then movieId else null end) {0} history,' \
                       'genre_sort(collect_list(case when label=1 then genres else null end) {1}) userGenres,' \
                       ' format_number(count(){1},4) ratingCount,' \
                       'format_number(avg(year) {1},4) avgYear,format_number(stddev(year) {1},4) stddevYear,' \
                       'format_number(avg(rating) {1},4) avgRating,format_number(stddev(rating) {1},4) stddevRating' \
                       ' from temp'.format(over_row5, over_row100)
    user_feature_df = common.select(spark, features, user_feature_sql)
    user_feature_df.show()
    return user_feature_df


if __name__ == '__main__':
    # 1、spark初始化
    spark = common.spark_init()

    spark.udf.register('array_2_vec', array_2_vec, VectorUDT())
    spark.udf.register('vector', Vectors.dense, VectorUDT())
    spark.udf.register('year', year, IntegerType())
    spark.udf.register('title', title, StringType())
    spark.udf.register('genre', genre, StringType())
    spark.udf.register('genre_sort', genre_sort,  ArrayType(StringType()))
    # array_2_vec_udf = F.udf(array_2_vec, VectorUDT())

    # 2、特征工程

    # # 读取csv
    # movie_df0 = spark.read.format('csv').option('header', 'true').load(movie_file)
    # # 保证数据格式(movieId int,title string,genre string)
    # """spark sql split '|'需要使用'\\\\|'"""
    # cast_sql = "select cast(movieId as int) movieId,title,explode(split(genres, '\\\\|')) as genre from temp"
    # movie_df = common.select(spark, movie_df0, cast_sql)
    # movie_df.show()

    # # 1、one_hot
    # one_hot(movie_df)

    # # 2、multi_hot
    # multi_hot(movie_df)

    ratings_df0 = spark.read.format('csv').option('header', 'true').load(ratings_file)
    # # 3、数值型特征的处理
    # numerical(ratings_df0)
    # 4、标签
    # label(ratings_df0)

    movie_df0 = spark.read.format('csv').option('header', 'true').load(movie_file)

    # 5、电影特征
    movie_feature_df = movie_features(movie_df0, ratings_df0)

    # 6、电影用户特征
    movie_user_feature_df = user_features(movie_feature_df)

    spark.stop()