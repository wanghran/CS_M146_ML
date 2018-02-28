import numpy as np
import matplotlib.pyplot as plt

x, y = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 1000).T
plt.title('Scatter Graph with mean [1,1] and individual variance is doubled')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y, 'o')
plt.savefig('Scatter Graph with mean [1,1] and individual variance is doubled.png', dpi=300)
plt.show()
