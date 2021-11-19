# 词向量
# 百度新闻数据集
# https://aistudio.baidu.com/aistudio/datasetdetail/24833


# 导入包
import pandas as pd
from gensim.models import Word2Vec
import os
import jieba

# 数据文件
data_dir = 'D:\\Users\\jiaxu\\data\\deepshare'
news_dir = os.path.join(data_dir, 'THUCNews')

# 模型文件
model_dir = 'D:\\Users\\jiaxu\\model\\deepshare'
model_file = os.path.join(model_dir, 'word2vec', 'word2vec_word_200')


def content_cut(x):
    x = jieba.lcut(x)
    x = " ".join(x)
    return x


def train():
    # 读取数据集
    train = pd.read_csv(os.path.join(news_dir, 'Train.tsv'), sep='\t', header=None, names=['label', 'text_a'])
    val = pd.read_csv(os.path.join(news_dir, 'Val.tsv'), sep='\t', header=None, names=['label', 'text_a'])
    print(train.head())


    train['text_a'] = train['text_a'].map(lambda x: content_cut(x))
    val['text_a'] = val['text_a'].map(lambda x: content_cut(x))

    df = pd.concat([train, val], axis=0)
    print(df.head())
    # 训练word2vec
    sentences = [document.split(' ') for document in df['text_a'].values]
    model = Word2Vec(sentences=sentences,
                     size=200,  # 维度
                     alpha=0.025,  # 默认
                     window=5,  # 默认
                     min_count=2,  # 2，3
                     sample=0.001,  #
                     seed=2018,  #
                     workers=11,  # 线程
                     min_alpha=0.0001,
                     sg=0,  # cbow
                     hs=0,  # 负采样
                     negative=5,  # 负采样个数
                     ns_exponent=0.75,
                     cbow_mean=1,  # 求和再取平均
                     iter=10,  # 10到20
                     compute_loss=True
                     )

    # 保存word2vec
    model.save(model_file)


def show():
    model = Word2Vec.load(model_file)
    help(model.most_similar)
    # 和这个单词最相似的单词
    model.most_similar("", topn=20)
    # 计算两个单词之间相似性
    model.similarity("816", "122")
    # 看词表
    model.wv.vocab.keys()


if __name__ == '__main__':
    # 生成模型
    # train()
    # 模型查看
    show()
    # 迭代模型


# # 迭代模型
#
# sentences_next = []
# for document in test['word_seg'].tolist():
#     sentences_next.append(document.split(" "))
#
# model.train(sentences=sentences_next, total_examples=model.corpus_count, epochs=model.iter)
#
# model.save()
