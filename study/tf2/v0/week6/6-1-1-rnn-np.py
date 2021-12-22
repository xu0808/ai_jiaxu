import numpy as np


class RNN:
    def __init__(self, word_dim, hidden_dim=100, output_dim=50):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (word_dim, hidden_dim))  # d*h
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))  # h*h
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, output_dim))  # h*q

    def forward_propagation(self, x):
        # train steps
        T = x.shape[1]
        N = x.shape[0]
        # hidden states 初始化为0
        s = np.zeros((N, T, self.hidden_dim))
        # output zeros
        o = np.zeros((N, T, self.output_dim))
        # for each time in step:
        for t in range(T):
            s[:, t, :] = np.tan(x[:, t, :].dot(self.U) + s[:, t - 1, :].dot(self.W))  # n*h
            o[:, t, :] = self.softmax(s[:, t, :].dot(self.V))  #
        return [o, s]

    def softmax(self, x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x


x = np.random.uniform(-0.5, 0.5, (10, 100, 20))
rnn = RNN(20)
o_, s_ = rnn.forward_propagation(x)
print("o_.shape = {}".format(o_.shape))
print("s_.shape = {}".format(s_.shape))

s = np.zeros((2, 3, 3))
print(s[:-1:])

