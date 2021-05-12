import matplotlib.pyplot as plt
import numpy as np

label = 'b={:.1f}'
w1 = 3.0
b1 = -8
b2 = 0
b3 = 8
l1 = label.format(b1)
l2 = label.format(b2)
l3 = label.format(b3)
x = np.arange(-10, 10, 0.1)
for b, l in zip([b1, b2, b3], [l1, l2, l3]):
    f = 1 / (1 + np.exp(-x * w1 - b))
    plt.plot(x, f, label=l)
plt.legend(loc=2)
plt.show()