import numpy as np
from math import sqrt
from collections import namedtuple
from time import clock
import random


# kd-tree每个结点中主要包含的数据结构如下
class KdNode(object):
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  # k维向量节点(k维空间中的一个样本点)
        self.split = split  # 整数（进行分割维度的序号）
        self.left = left  # 该结点分割超平面左子空间构成的kd-tree
        self.right = right  # 该结点分割超平面右子空间构成的kd-tree


class KdTree(object):
    def __init__(self, data):
        dimension = len(data[0])  # 数据维度

        def create_node(split, data_set):  # 按第split维划分数据集exset创建KdNode
            if not data_set:  # 数据集为空
                return None
            # key参数的值为一个函数，此函数只有一个参数且返回一个值用来进行比较
            # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为需要获取的数据在对象中的序号
            # data_set.sort(key=itemgetter(split)) # 按要进行分割的那一维数据排序
            data_set.sort(key=lambda x: x[split])
            split_pos = len(data_set) // 2  # //为Python中的整数除法
            median = data_set[split_pos]  # 中位数分割点
            split_next = (split + 1) % dimension  # cycle coordinates

            # 递归的创建kd树
            return KdNode(median, split,
                          create_node(split_next, data_set[:split_pos]),  # 创建左子树
                          create_node(split_next, data_set[split_pos + 1:]))  # 创建右子树

        self.root = create_node(0, data)  # 从第0维分量开始构建kd树,返回根节点


# KDTree的前序遍历
def preorder(root):
    print(root.dom_elt)
    if root.left:  # 节点不为空
        preorder(root.left)
    if root.right:
        preorder(root.right)

    # create a dataset which contains 4 samples with 2 classes


# 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
result = namedtuple("Result_tuple", "nearest_point  nearest_dist  nodes_visited")


def find_nearest(tree, point):
    dimension = len(point)  # 数据维度

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * dimension, float("inf"), 0)  # python中用float("inf")和float("-inf")表示正负无穷

        nodes_visited = 1

        s = kd_node.split  # 进行分割的维度
        pivot = kd_node.dom_elt  # 进行分割的“轴”

        if target[s] <= pivot[s]:  # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            nearer_node = kd_node.left  # 下一个访问节点为左子树根节点
            further_node = kd_node.right  # 同时记录下右子树
        else:  # 目标离右子树更近
            nearer_node = kd_node.right  # 下一个访问节点为右子树根节点
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)  # 进行遍历找到包含目标点的区域

        nearest = temp1.nearest_point  # 以此叶结点作为“当前最近点”
        dist = temp1.nearest_dist  # 更新最近距离

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist  # 最近点将在以目标点为球心，max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s])  # 第s维上目标点与分割超平面的距离
        if max_dist < temp_dist:  # 判断超球体是否与超平面相交
            return result(nearest, dist, nodes_visited)  # 不相交则可以直接返回，不用继续判断

        # ----------------------------------------------------------------------
        # 计算目标点与分割点的欧氏距离
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))

        if temp_dist < dist:  # 如果“更近”
            nearest = pivot  # 更新最近点
            dist = temp_dist  # 更新最近距离
            max_dist = dist  # 更新超球体半径

        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(further_node, target, max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:  # 如果另一个子结点内存在更近距离
            nearest = temp2.nearest_point  # 更新最近点
            dist = temp2.nearest_dist  # 更新最近距离

        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))  # 从根节点开始递归


# classify using kNN (k Nearest Neighbors )
# Input:      newInput: 1 x N
#             dataSet:  M x N (M samples N features)
#             labels:   1 x M
#             k: number of neighbors to use for comparison
# Output:     the most popular class label
def knn_classify(new_input, data_set, labels, k):
    data_set = np.array(data_set)
    num_samples = data_set.shape[0]  # shape[0] stands for the num of row

    # step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy num_samples rows for dataSet
    diff = np.tile(new_input, (num_samples, 1)) - data_set  # Subtract element-wise
    squared_diff = diff ** 2  # squared for the subtract
    squared_dist = np.sum(squared_diff, axis=1)  # sum is performed by row
    distance = squared_dist ** 0.5

    # step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    sorted_dist_indices = np.argsort(distance)
    class_count = {}  # define a dictionary (can be append element)
    return result(data_set[sorted_dist_indices[0]],
                  distance[np.searchsorted(sorted_dist_indices, 0)], 8)
    # for i in range(k):
    #     # step 3: choose the min k distance
    #     vote_label = labels[np.searchsorted(sorted_dist_indices, i)]
    #
    #     # step 4: count the times labels occur
    #     # when the key voteLabel is not in dictionary class_count, get()
    #     # will return 0
    #     class_count[vote_label] = class_count.get(vote_label, 0) + 1
    #
    #     # step 5: the max voted class will return
    # max_count = list(class_count.popitem())
    # for key, value in class_count.items():
    #     if value > max_count[1]:
    #         max_count[1] = value
    #         max_count[0] = key
    #
    # return max_count[0],


# 产生一个k维随机向量，每维分量值在0~1之间
def random_point(k):
    return [random.random() for _ in range(k)]


def random_lable(k):
    return [chr(random.randint(65, 90)) for _ in range(k)]


# 产生n个k维随机向量
def random_points(k, n):
    return [random_point(k) for _ in range(n)]


if __name__ == "__main__":
    N = 400000
    random_datasets = random_points(3, N)
    random_lables = random_lable(N)
    kd2 = KdTree(random_datasets)  # 构建包含四十万个3维空间样本点的kd树
    t0 = clock()
    ret2 = find_nearest(kd2, [0.1, 0.5, 0.8])  # 四十万个样本点中寻找离目标最近的点
    t1 = clock()
    print("time: ", t1 - t0, "s")
    print(ret2)
    ret3 = knn_classify([0.1, 0.5, 0.8], random_datasets, random_lables, 1)
    t2 = clock()
    print("time: ", t2 - t0, "s")
    print(ret3)
