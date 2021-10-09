#!/usr/bin/env python
# coding: utf-8
# 图神经网络特征工程-再整理
"""
1、提取行为序列
2、计算跳转概率
3、随机游走生产序列样本
4、Word2Vec训练
5、预测可以通过Word2Vec模型
  也可以通过LSH模型
"""

# 保证aicloud环境可以识别自定义包
import os
import sys

if 'dev' != os.getenv('MY_NAMESPACE', 'dev'):
    sys.path.append('/var/ai-cloud/project/rec_feature_pyspark/python')

from pyspark.sql.types import *
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.mllib.feature import Word2Vec
from pyspark.ml.linalg import Vectors
import random
import numpy as np
from feature import common

# 每个向量大小
emb_size = 10
# 随机游走采样数量
sample_total = 10 # 20000
# 每个样本大小
sample_size = 10

# 数据文件
if not os.path.exists(common.out_dir):
    os.makedirs(common.out_dir)
item2vec_file = os.path.join(common.out_dir, 'item2vecEmb.csv')
itemGraph_file = os.path.join(common.out_dir, 'itemGraphEmb.csv')
userEmb_file = os.path.join(common.out_dir, 'userEmb.csv')


def item_seq(movie_list):
    """
    2维数组取第二个字段拼接字符串
    :param
    [[1112484676, 29], [1112484819, 32], [1112486027, 2]]
    :return:
    ['29', '32', '2']
    ###!!! 整型数组输出数值为空，需要改成字符型
    """
    return [str(m[1]) for m in movie_list]


def seqs_pair(seqs):
    """
    序列成对拆分
    :param
    ['29', '32', '2']
    :return:
    [['29', '32'],['32', '2']]
    """
    pairs = []
    pre_item = None
    for item in seqs:
        if pre_item:
            pairs.append((pre_item, item))
        pre_item = item
    return pairs


# 观影序列（实际使用好评序列）
def movie_seq(df_0):
    # 1、组内合并数组
    group_sql = 'select userId,collect_list(struct(timestamp,movieId)) array' \
                ' from temp where rating >= 3.5 group by userId'
    # rating gat
    group_df = common.select(spark, df_0, group_sql)
    group_df.show(20, False)

    # 2、数组排序合并
    # sort_array 默认第一个字段升序排列
    # [1094785598, 924], [1094785621, 919], [1094785709, 337]
    sort_sql = 'select item_seq(sort_array(array)) item_seq from temp'
    # rating gat
    sort_df = common.select(spark, group_df, sort_sql)
    sort_df.show(20, False)
    return sort_df


def train_item2vec(seqs_rdd, out_file):
    # 1、模型定义
    word2vec = Word2Vec().setVectorSize(emb_size) \
        .setWindowSize(5).setNumIterations(10)
    # 2、模型训练(只取行中的一列 Seq[String])
    word2vec_model = word2vec.fit(seqs_rdd)
    # 3、模型预测
    # 查找与item"158"最相似的20个item
    synonyms = word2vec_model.findSynonyms("158", 20)
    for synonym, cosineSimilarity in synonyms:
        # 相似电影id， 相似度系数
        print(synonym, cosineSimilarity)

    # 获取所有Embedding向量,写入文件
    # key->movie_id, value->emb_list
    emb_dict = word2vec_model.getVectors()
    with open(out_file, 'w') as f:
        for movie_id, emb_list in emb_dict.items():
            vectors = " ".join([str(emb) for emb in emb_list])
            f.write(movie_id + ":" + vectors + "\n")

    emb_lsh(emb_dict)
    return word2vec_model


def emb_lsh(emb_dict):
    # 1、数据转换
    movie_emb_seq = []
    for key, emb_list in emb_dict.items():
        # float64
        emb_list = [np.float64(emb) for emb in emb_list]
        # vectors.dense
        movie_emb_seq.append((key, Vectors.dense(emb_list)))

    movie_emb_df = spark.createDataFrame(movie_emb_seq).toDF("movieId", "emb")

    # 2、局部敏感hash模型
    bucket_lsh = BucketedRandomProjectionLSH(inputCol="emb", outputCol="bucketId",
                                             bucketLength=0.1, numHashTables=3)
    bucket_model = bucket_lsh.fit(movie_emb_df)

    # 3、打印训练结果
    result = bucket_model.transform(movie_emb_df)
    print("movieId, emb, bucketId schema:")
    result.printSchema()
    print("movieId, emb, bucketId gat result:")
    result.show(10, truncate=False)

    # 4、近邻搜索top5
    print("Approximately searching for 5 nearest neighbors of the sample embedding:")
    sample_emb = Vectors.dense(0.795, 0.583, 1.120, 0.850, 0.174, -0.839, -0.0633, 0.249, 0.673, -0.237)
    bucket_model.approxNearestNeighbors(movie_emb_df, sample_emb, 5).show(truncate=False)


def trans_matrix(seqs_df):
    """
    转移概率矩阵
    """
    # 1、所有组合清单
    pair_sql = 'select pair[0] as pair_0,pair[1] as pair_1 from temp lateral view explode(seqs_pair(item_seq)) as pair'
    pair_df = common.select(spark, seqs_df, pair_sql)
    pair_df.show(20, False)
    # 总对数
    pair_total = pair_df.count()

    # 2、每个组合次数
    pair_count_sql = 'select pair_0,pair_1,count(1) num from temp group by pair_0,pair_1'
    pair_count_df = common.select(spark, pair_df, pair_count_sql)
    pair_count_df.show(20, False)

    # 3、head概率
    head_per_sql = 'select pair_0,sum(num) num,cast(sum(num)/%d as decimal(6,5)) as per ' \
                   'from temp group by pair_0' % pair_total
    head_per_df = common.select(spark, pair_count_df, head_per_sql)
    head_per_df.show(20, False)
    head_per_df.createOrReplaceTempView('head')

    # 4、转移概率
    pair_per_sql = 'select p.pair_0,p.pair_1,cast(p.num/h.num as decimal(6,5)) as per ' \
                   'from temp p left join head h on p.pair_0=h.pair_0'
    pair_per_df = common.select(spark, pair_count_df, pair_per_sql)
    pair_per_df.show(20, False)
    return pair_per_df, head_per_df


def one_random_walk(pair_per_dict, head_per_list):
    sample = []
    # 1、遍历头分布取首个物品
    random_d = random.random()
    head_item = None
    acc_prob = 0.0
    for row in head_per_list:
        item, prob = row[0], row[1]
        acc_prob += np.float(prob)
        # 概率之和达到随机数，取得首个物品
        if acc_prob >= random_d:
            head_item = item
            break
    sample.append(head_item)

    # 2、使用转移概率逐个添加序列
    cur_item = head_item
    i = 1
    while i < sample_size:
        random_d = random.random()
        acc_prob = 0.0
        for item, prob in pair_per_dict[head_item]:
            acc_prob += np.float(prob)
            # 概率之和达到随机数，取得当前物品
            if acc_prob >= random_d:
                cur_item = item
                break
        sample.append(cur_item)
        i += 1
    return sample


def random_walk(pair_per_dict, head_per_list):
    samples = []
    for i in range(sample_total):
        samples.append(one_random_walk(pair_per_dict, head_per_list))
    return samples


def movie_graph_emb(seqs_df, out_file):
    # 1、计算迁移矩阵
    # 迁移概率、起点概率
    pair_per_df, head_per_df = trans_matrix(seqs_df)
    # pair_0, pair_1, per
    pair_per_dict = {}
    for row in pair_per_df.rdd.collect():
        pair_0, pair_1, per = row[0], row[1], row[2]
        if pair_0 not in pair_per_dict:
            pair_per_dict[pair_0] = []
        pair_per_dict[pair_0].append([pair_1, per])

    # pair_0,num,per
    head_per_list = head_per_df.rdd.collect()
    # 2、随机游走生成序列
    samples = random_walk(pair_per_dict, head_per_list)
    # 3、序列样本生成向量
    rdd = spark.sparkContext.parallelize(samples)
    train_item2vec(rdd, out_file)


def user_emb(model, out_file):
    # 1、计算结果转换df
    vec_list = []
    for key, value in model.getVectors().items():
        vec_list.append((key, list(value)))
    fields = [
        StructField('movieId', StringType(), False),
        StructField('emb', ArrayType(FloatType()), False)
    ]
    schema = StructType(fields)
    vec_df = spark.createDataFrame(vec_list, schema=schema)
    # 2、计算结果转换df
    user_emb_df = ratings_df.join(vec_df, on='movieId', how='inner')
    user_emb_list_sql = 'select userId,collect_list(emb) emb_list from temp grop by userId'
    user_emb_list_df = common.select(spark, user_emb_df, user_emb_list_sql)
    user_emb_list_df.show(20, False)

    with open(out_file, 'w') as f:
        for row in user_emb_list_df.rdd.collect():
            vectors = " ".join([str(emb) for emb in row[1]])
            f.write(row[0] + ":" + vectors + "\n")


if __name__ == '__main__':
    # 1、spark初始化
    spark = common.spark_init()
    # 注册udf(定义输出类型)
    spark.udf.register('item_seq', item_seq, ArrayType(StringType()))
    spark.udf.register('seqs_pair', seqs_pair, ArrayType(ArrayType(StringType())))

    # 2、评分数据
    # 取3个用户快速测试
    ratings_df = common.read_ratings(spark)
    ratings_df.show()
    ratings_df.persist()

    # 3、观影序列（实际使用好评序列）
    movie_seq_df = movie_seq(ratings_df)
    movie_seq_df.persist()
    # 4、物品向量化(embedding)
    model_0 = train_item2vec(movie_seq_df.rdd.map(lambda x: x[0]), out_file=item2vec_file)
    # 5、电影序列图向量化(graph embedding)
    movie_graph_emb(movie_seq_df, out_file=itemGraph_file)
    # 6、序列图向量化(graph embedding)
    user_emb(model_0, out_file=userEmb_file)

