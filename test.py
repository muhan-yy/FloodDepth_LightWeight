import numpy as np
from scipy.ndimage import binary_erosion, label
import os
import cv2

# 验证edge和image是否匹配

edge_txt = './oriData/labels_train1/141057_3267045283.txt'
imageOri = './oriData/images/141057_3267045283.jpg'
image = cv2.imread(imageOri)
height, width, _ = image.shape
edges = []
with open(edge_txt, 'r') as f:
    for row in f.readlines():
        edge_info = list(map(float, row.strip().split(' ')[1:]))
        edges = [edge_info[i:i+2] for i in range(0, len(edge_info), 2)]
edges = np.array(edges)
edges[:,0] *= height
edges[:,1] *= width
edges = np.round(edges).astype(int)
image[edges[:,0], edges[:,1]] = [0,255,0]
cv2.imshow('Modified Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# COCO8seg边缘数据是基于原始图片获取的 -> 说明对应函数没有问题
# COCO8seg边缘数据是根据SAM生成的npy格式的mask获取的
# 根据SAM生成的txt格式的mask存在问题 -> 多测几个图片看看到底有没有问题
# 所以不将SAM生成的mask存储为txt，直接将获取的mask存储成txt格式的COCO8seg边缘数据
# 而且只需要跑人工check之后的类别图片




