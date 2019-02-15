#coding:utf-8


import numpy as np

from kmeans import kmeans, avg_iou, load_dataset,visualize_data

ANNOTATIONS_PATH = "D:/DATASETS/VOC2012/VOCtrainval/VOC2007/Annotations"
CLUSTERS = 5

data = load_dataset(ANNOTATIONS_PATH)

visualize_data(data)

out = kmeans(data, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}".format(out))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))