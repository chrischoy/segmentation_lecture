from sklearn.cluster import MeanShift

import utils
import matplotlib.pyplot as plt

# Get vectorized image
feat, im = utils.load_bilateral_image()
H, W = im.shape[:2]

ms = MeanShift(bandwidth=1, bin_seeding=True)
ms.fit(feat)
labels = ms.labels_

plt.subplot(1, 2, 1)
plt.imshow(im)
plt.subplot(1, 2, 2)
plt.imshow(labels.reshape(H, W))
plt.show()
