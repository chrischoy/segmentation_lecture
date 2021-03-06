from PIL import Image
import numpy as np

import matplotlib.pyplot as plt


def sample_mm_gaussian(N=1000, K=10, mode_sigma=5):
  """
  Sample n points from k multi modal gaussian distribution with covariance
  [[1, 0], [0, 1]].
  """
  # Stick breaking for weight
  def stick_break(N):
    n = int(np.random.rand(1) * N)
    return n, N - n

  def sample(n):
    mean = mode_sigma * np.random.randn(1, 2)
    sigma = np.random.rand(1) * 2
    return sigma * np.random.randn(n, 2) + mean

  rem = N
  xs = []
  for k in range(K - 1):
    n, rem = stick_break(rem)
    x = sample(n)
    xs.append(x)

  x = sample(rem)
  xs.append(x)
  return np.concatenate(xs, 0)


def load_image(filename='stanford.jpg'):
  im = np.array(Image.open('images/' + filename))
  if im.ndim > 2 and im.shape[2] > 3:
    im = im[..., :3]
  return im


def whiten(im):
  H, W, C = im.shape
  # Flatten
  whitened_im = im.reshape(-1, C).astype('float')
  # Whiten the feature
  whitened_im -= np.mean(whitened_im, 0)
  whitened_im /= np.sqrt(np.mean(whitened_im ** 2, 0))
  return whitened_im.reshape(H, W, C)


def load_bilateral_image():
  im = load_image()
  H, W = im.shape[:2]
  y, x = np.mgrid[:H, :W]
  feat = np.concatenate(
      (y[..., np.newaxis], x[..., np.newaxis], im), 2).astype('float')
  return feat, im


if __name__ == '__main__':
  X = sample_mm_gaussian()
  plt.scatter(X[:, 0], X[:, 1])
  plt.show()
