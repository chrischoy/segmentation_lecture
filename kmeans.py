import numpy as np
from sklearn.cluster import KMeans

import utils
import matplotlib.pyplot as plt


feat, im = utils.load_bilateral_image()
H, W = im.shape[:2]

# Flatten
feat = feat.reshape(-1, 5)

# Whiten the feature
feat -= np.mean(feat, 0)
feat /= np.sqrt(np.mean(feat ** 2, 0))

# K-means
km = KMeans(10)
km.fit(feat)
labels = km.labels_

plt.imshow(labels.reshape(H, W))
plt.show()
