# numpy实现卷积
import numpy as np

"""
TensorFlow有两种数据格式NHWC和NCHW，默认的数据格式是NHWC，可以通过参数data_format指定数据格式。
这个参数规定了 input Tensor 和 output Tensor 的排列方式。
设置为 “NHWC” 时，排列顺序为 [batch, height, width, channels]
设置为 “NCHW” 时，排列顺序为 [batch, channels, height, width]
"""


def conv_numpy(x, w, b, pad, strides):
    out = None

    N, H, W, C = x.shape
    F, HH, WW, C = w.shape
    # 填充
    X = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')

    Hn = 1 + int((H + 2 * pad - HH) / strides[0])
    Wn = 1 + int((W + 2 * pad - WW) / strides[1])

    out = np.zeros((N, Hn, Wn, F))

    for n in range(N):
        for m in range(F):
            for i in range(Hn):
                for j in range(Wn):
                    # 当前第n个样本，第m个输出通道，（i,j）位置的值为：
                    # 起止点（i * strides[0]，j * strides[1]）, 高HH, 宽WW
                    (h0, w0) = (i * strides[0], j * strides[1])
                    (h1, w1) = (h0 + HH, w0 + WW)
                    # 当前卷积块（h0, w0, h1, w1）所有通道
                    data = X[n, h0:h1, w0:w1, :].reshape(1, -1)
                    # 第m个过滤器权重
                    filt = w[m].reshape(-1, 1)
                    # 当前点卷积结果为:点乘+偏值
                    out[n, i, j, m] = data.dot(filt) + b[m]

    return out


inputs_ = np.random.random((10, 28, 28, 3)).astype(np.float32)
w = np.random.random((6, 3, 3, 3)).astype(np.float32)
b = np.random.random((6,)).astype(np.float32)

a = conv_numpy(inputs_, w=w, b=b, pad=1, strides=(1, 1))
print('a.shape = ', a.shape)
