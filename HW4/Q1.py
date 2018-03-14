import matplotlib.pyplot as plt


class point:

    def __init__(self, x, y, label, weight):
        self.x = x
        self.y = y
        self.label = label
        self.weight = weight


def plot():

    x = [0, 1, -2, -1, 9, -7]
    y = [8, 4, 1, 13, 11, -1]
    x2 = [3, 12, -3, 5]
    y2 = [7, 7, 12, 9]
    plt.plot(x, y, 'o')
    plt.plot(x2, y2, 'x')
    plt.show()

data = []

data.append(point(0, 8, False, 0.0625))
data.append(point(1, 4, False, 0.0625))
data.append(point(3, 7, True, 0.0625))
data.append(point(-2, 1, False, 0.0625))
data.append(point(-1, 13, False, 0.0625))
data.append(point(9, 11, False, 0.25))
data.append(point(12, 7, True, 0.0625))
data.append(point(-7, -1, False, 0.0625))
data.append(point(-3, 12, True, 0.25))
data.append(point(5, 9, True, 0.0625))

error = {}
for i in range(-8, 14):
    e = 0
    for j in range(10):
        if data[j].x > i and data[j].label is not True:
            e += data[j].weight
        elif data[j].x <= i and data[j].label is not False:
            e += data[j].weight
    error[i] = e

print(error)




plot()

