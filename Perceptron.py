import numpy as np
import random
import matplotlib.pyplot as plt
import time


def sign(v):
    if v > 0:
        return 1
    else:
        return -1


def normal_perceptron(train_num, train_datas, lr):
    w = [0] * (len(train_datas[0]) - 1)
    b = 0
    train_datas = np.array(train_datas)
    y = train_datas[:, -1]
    x = train_datas[:, 0:-1]
    for i in range(train_num):
        i = random.randint(0, len(train_datas) - 1)
        if y[i] * (np.matmul(w, x[i].T) + b) <= 0:
            w += y[i] * x[i] * lr
            b += y[i] * lr
        l = 0
        # for j in range(len(train_datas) - 1):
        #     l += (1 - sign(y[i] * (np.matmul(w, x[i].T) + b)))
        # if l == 0:
        #     return w, b
    return w, b


def duality_perceptron(train_num, train_datas, lr):
    w = [0.0] * (len(train_datas[0]) - 1)
    b = 0
    train_datas = np.array(train_datas)
    y = train_datas[:, -1]
    x = train_datas[:, :-1]
    gram = np.matmul(x, x.T)
    datas_len = len(train_datas)
    alpha = [0] * datas_len
    for n in range(train_num):
        temp = 0
        i = random.randint(0, datas_len - 1)
        for j in range(datas_len - 1):
            temp += alpha[j] * y[j] * gram[i, j]
        temp += b

        if y[i] * temp <= 0:
            alpha[i] += lr
            b += lr * y[i]
    for i in range(datas_len - 1):
        w += alpha[i] * y[i] * x[i]
    return w, b, alpha, gram


def plot_points(train_datas, w, b):
    plt.figure()
    x1 = np.linspace(0, 8, 100)
    x2 = (-b - w[0] * x1) / (w[1] + 1e-10)
    plt.plot(x1, x2, color='yellow', label='y1 data')
    datas_len = len(train_datas)
    for i in range(datas_len):
        if (train_datas[i][-1] == 1):
            plt.scatter(train_datas[i][0], train_datas[i][1], color='black')
        else:
            plt.scatter(train_datas[i][0], train_datas[i][1], color='red', marker='x', s=50)
    plt.show()


if __name__ == '__main__':
    train_data1 = [[1, 3, 1], [2, 2, 1], [3, 8, 1], [2, 6, 1]]  # 正样本
    train_data2 = [[2, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1]]  # 负样本
    train_datas = train_data1 + train_data2  # 样本集
    time_begin = time.time()
    # w, b, alpha, gram = duality_perceptron(500, train_datas, 0.01)
    w, b = normal_perceptron(500, train_datas, 0.01)
    time_end = time.time()
    print(str(time_end - time_begin))
    plot_points(train_datas, w, b)
