from sklearn.cluster import MeanShift

from utils import load_bilateral_image, whiten
import matplotlib.pyplot as plt

# Get vectorized image
feat, im = load_bilateral_image()
H, W = im.shape[:2]
feat = whiten(feat)

ms = MeanShift(bandwidth=1, bin_seeding=True)
ms.fit(feat.reshape(-1, feat.shape[2]))
labels = ms.labels_

plt.subplot(1, 2, 1)
plt.imshow(im)
plt.subplot(1, 2, 2)
plt.imshow(labels.reshape(H, W))
plt.show()
