# 分析数据（统计分类情况）
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_rows = 20
sns.set(style="darkgrid")

data_dir = 'D:\\Users\\jiaxu\\data\\deepshare'
quick_draw_dir = os.path.join(data_dir, 'quick_draw')

# 数据集介绍
# quick draw中的20个分类，包括
classfication = ['alarm clock', 'eiffel tower', 'angel', 'ant', 'car',
                 'carrot', 'helmet', 'helicopter', 'ladder', 'lightning',
                 'mosquito', 'pillow', 'rain', 'radio', 'vase',
                 'umberlla', 'train', 'watermelon', 'wheel', 'stairs'
                 ]

# 读取所有标注
labels = []
for nzp in os.listdir(quick_draw_dir):
    if nzp.startswith('train_batch'):
        data = np.load(os.path.join(quick_draw_dir, nzp))
        labels = labels + data['labels'].tolist()

print("len(labels) = {}".format(len(labels)))

# 转换成分类名称
words = [classfication[label] for label in labels]

# list转df
df = pd.DataFrame({'word':words})
# 安装数量统计
count_gp = df.groupby(['word']).size().reset_index(name='count').sort_values('count', ascending=False)
print('count_gp = ', count_gp)
top_10 = count_gp[:10]
total = count_gp.shape[0]
bottom_10 = count_gp[total-10:total]
print('top_10 = ', top_10)
print('bottom_10 = ', bottom_10)

# top10直方图
ax_t10 = sns.barplot(x="word", y="count", data=top_10, palette="coolwarm",ci=500)
ax_t10.set_xticklabels(ax_t10.get_xticklabels(), rotation=40, ha="right")
plt.show()

# last10直方图
ax_b10 = sns.barplot(x="word", y="count", data=bottom_10, palette="BrBG")
ax_b10.set_xticklabels(ax_b10.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()