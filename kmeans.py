from sklearn.cluster import KMeans

from utils import load_bilateral_image, whiten
import matplotlib.pyplot as plt


# Get vectorized image
feat, im = load_bilateral_image()
H, W = im.shape[:2]
feat = whiten(feat)

# K-means
km = KMeans(10)
km.fit(feat.reshape(-1, feat.shape[2]))
labels = km.labels_

plt.subplot(1, 2, 1)
plt.imshow(im)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(labels.reshape(H, W))
plt.axis('off')
plt.show()
