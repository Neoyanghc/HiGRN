import numpy as np
import pandas as pd
import time
import datetime
import math
from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt


def distance_adj_mx():
    np.load(adj_mx.npy)
    # """
    # 使用带有阈值的高斯核计算邻接矩阵的权重，如果有其他的计算方法，可以覆盖这个函数,
    # 公式为：$ w_{ij} = \exp \left(- \\frac{d_{ij}^{2}}{\sigma^{2}} \\right) $, $\sigma$ 是方差,
    # 小于阈值`weight_adj_epsilon`的值设为0：$  w_{ij}[w_{ij}<\epsilon]=0 $
    #
    # Returns:
    #     np.ndarray: self.adj_mx, N*N的邻接矩阵
    # """
    #
    # distances = self.adj_mx[~np.isinf(self.adj_mx)].flatten()
    # std = distances.std()
    # self.adj_mx = np.exp(-np.square(self.adj_mx / std))
    # self.adj_mx[self.adj_mx < self.weight_adj_epsilon] = 0


def region_adj_mx():
    # 根据地理位置将所有浮标划分到不同到region上
    # 23°26′ N（北回归线）到23°26′ S（南回归线）为热带
    # 北纬（南纬）23°26′到北纬（南纬）66°34′（北极圈、南极圈）为南 北温带
    # 北纬（南纬）66°34′ 到北纬（南纬）90度为南 北寒带
    data = pd.read_csv('/root/Ocean_sensor_model/raw_data/Ocean_sensor_2022_2/Ocean_sensor_2022_2.geo')
    class_list = []
    for index, row in data.iterrows():
        lat = float(row['coordinates'].split(',')[0][1:])
        class_index = 0
        if -23.26 < lat < 23.26:
            class_index = 0
        if 23.26 < lat < 66.34:
            class_index = 1
        if 66.34 < lat < 90:
            class_index = 2
        if -66.34 < lat < -23.26:
            class_index = 3
        if -90 < lat < -66.34:
            class_index = 4
        class_list.append(class_index)


def similarity_adj_mx():
    # 根据各个点之间的相似性建立多图，多通道进行预测
    data = pd.read_csv('/root/Ocean_sensor_model/raw_data/Ocean_sensor_2022_2/Ocean_sensor_2022_2.dyna')
    data_group = [j.to_list() for i,j in data.groupby('entity_id')['temp']]
    array = np.array(data_group)
    array = np.resize(array, [420, -1])
    # array是数据数组
    s = cosine_similarity(array)
    zero = np.zeros(shape=[array.shape[0],array.shape[0]])
    knn = 5
    for i in range(len(s)):
        ind = np.argpartition(s[i], -knn)[-knn:]
        zero[i][ind] = 1
    # s = cosine_similarity(array)
    # distances = s[~np.isinf(s)].flatten()
    # std = distances.std()
    # s = np.exp(-np.square(s / std))
    # s[s < 0.2] = 0
    print(s)


if __name__ == '__main__':
    # similarity_adj_mx()
    region_adj_mx()
    


