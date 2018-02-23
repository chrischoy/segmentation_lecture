import numpy as np
import matplotlib.pyplot as plt

from utils import load_image, whiten


def X(im, i, j):
  """ Return 3x3 image features. Assume that 0 < i < H, 0 < j < W"""
  return im[i - 1:i + 2, j - 1:j + 2].reshape(-1, 3)


def loss_and_grad(w, y, x, lamb=0.0001):
  # Type casting
  x = x.reshape(-1)

  wdotx = w.dot(x)
  grad = lamb * w
  if 1 > y * wdotx:
    grad -= (y * x)
  loss = max(0, 1 - y * wdotx) + lamb * np.sum(w ** 2)
  return loss, grad


def predict(wim, w):
  H, W = wim.shape[:2]
  pred = np.zeros((H, W))

  for i in range(1, H - 1):
    for j in range(1, W - 1):
      pred[i, j] = w.dot(X(wim, i, j).reshape(-1))

  return pred


im = load_image('cat1.jpg')
im2 = load_image('cat2.jpg')
y = load_image('cat1_label.png').astype('int')
y = y * 2 - 1  # {-1, 1}^{H \times W}
H, W = im.shape[:2]
MAX_ITER = 2000

# Whiten the image
whitened_im = whiten(im)
whitened_im2 = whiten(im2)

# Define weight
w = np.random.rand(27) / 27

for curr_iter in range(MAX_ITER + 1):
  # Get random image patch
  i = np.random.randint(1, H - 1)
  j = np.random.randint(1, W - 1)

  # Get loss and gradient
  loss, grad = loss_and_grad(w, y[i, j], X(whitened_im, i, j))
  w -= 0.001 * grad

  if curr_iter % 50 == 0:
    print("Iter: {}\tLoss: {:.4}".format(curr_iter, loss))

  if curr_iter % 1000 == 0:
    pred = predict(whitened_im2, w)
    plt.imshow(pred)
    plt.clim(-1, 1)
    plt.axis('off')
    plt.show()
