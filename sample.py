import numpy as np


def sample_mm_gaussian(N=10000, K=5):
  """
  Sample n points from k multi modal gaussian distribution with covariance [[1, 0], [0, 1]].
  """
  # Stick breaking for weight
  def stick_break(N):
    n = int(np.random.rand(1) * N)
    return n, N - n

  def sample(n):
    mean = 3 * np.random.randn(1, 2)
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


if __name__ == '__main__':
  X = sample_mm_gaussian()
