import numpy as np
from sklearn.cluster import KMeans

import utils
import matplotlib.pyplot as plt


# Get vectorized image
feat, im = utils.load_bilateral_image()
H, W = im.shape[:2]

# K-means
km = KMeans(10)
km.fit(feat)
labels = km.labels_

plt.subplot(1, 2, 1)
plt.imshow(im)
plt.subplot(1, 2, 2)
plt.imshow(labels.reshape(H, W))
plt.show()
