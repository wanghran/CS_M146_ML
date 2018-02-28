import numpy as np

w, v = np.linalg.eig([[1, 0], [1, 3]])
n = np.argmax(w)
print(v[:, n])

