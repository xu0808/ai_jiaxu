"""
Created on July 13, 2020

dataset：criteo dataset sample
features：
- Label - Target variable that indicates if an ad was clicked (1) or not (0).
- I1-I13 - A total of 13 columns of integer features (mostly count features).
- C1-C26 - A total of 26 columns of categorical features.
The values of these features have been hashed onto 32 bits for anonymization purposes.

@author: Ziyao Geng(zggzy1996@163.com)
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split

from utils import sparseFeature


def create_criteo_dataset(file, embed_dim=8, read_part=True, sample_num=200, test_size=0.2):
    """
    a example about creating criteo dataset
    :param file: dataset's path
    :param embed_dim: the embedding dimension of sparse features
    :param read_part: whether to read part of it
    :param sample_num: the number of instances if read_part is True
    :param test_size: ratio of test dataset
    :return: feature columns, train, test
    """

    if read_part:
        # 部分读取【iterator=True】
        data_df_0 = pd.read_csv(file, iterator=True)
        data_df = data_df_0.get_chunk(sample_num)

    else:
        data_df = pd.read_csv(file)
    # 字符特征
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    # 默认取'-1'
    data_df[sparse_features] = data_df[sparse_features].fillna('-1')
    # 数值特征
    dense_features = ['I' + str(i) for i in range(1, 14)]
    # 默认取0
    data_df[dense_features] = data_df[dense_features].fillna(0)
    # 全部特征
    features = sparse_features + dense_features

    # K-bins离散化（分箱）
    est = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    data_df[dense_features] = est.fit_transform(data_df[dense_features])

    # 标签顺序字典
    for feat in sparse_features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    # ==============Feature Engineering===================

    # ====================================================
    feature_columns = [sparseFeature(feat, int(data_df[feat].max()) + 1, embed_dim=embed_dim)
                        for feat in features]
    train, test = train_test_split(data_df, test_size=test_size)

    train_X = train[features].values.astype('int32')
    train_y = train['label'].values.astype('int32')
    test_X = test[features].values.astype('int32')
    test_y = test['label'].values.astype('int32')

    return feature_columns, (train_X, train_y), (test_X, test_y)