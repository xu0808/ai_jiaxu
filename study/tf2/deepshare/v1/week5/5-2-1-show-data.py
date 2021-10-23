# 查看图片
import os
import numpy as np
import cv2

data_dir = 'D:\\Users\\jiaxu\\data\\deepshare'
quick_draw_dir = os.path.join(data_dir, 'quick_draw')

# 数据集介绍
# quick draw中的20个分类，包括
classfication = ['alarm clock', 'eiffel tower', 'angel', 'ant', 'car',
                 'carrot', 'helmet', 'helicopter', 'ladder', 'lightning',
                 'mosquito', 'pillow', 'rain', 'radio', 'vase',
                 'umberlla', 'train', 'watermelon', 'wheel', 'stairs'
                 ]

# 每个batch中有200，000张图片
# 这样来获得data和label
x = np.load(os.path.join(quick_draw_dir, 'train_batch_1.npz'))
data = x['gat']
label = x['labels']

# 这样来查看某张图片
import matplotlib.pyplot as plt

data = x['gat']
img = np.reshape(data, [-1, 28, 28])[0]
plt.imshow(img)
plt.show()

# cv2放大图片
img_128 = cv2.resize(img, (128, 128))
plt.imshow(img_128)
plt.show()
