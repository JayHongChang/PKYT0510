import matplotlib.pyplot as plt
import numpy as np

# y= ax+b
b = 5
print(b)
a = np.linspace(5, -5, 11)
x = np.arange(-10, 10, 0.1)
#print(x)

for a1 in a:
    y = a1 * x + b
    plt.plot(x, y, label=f"y={a1}x+{b}")
    plt.legend(loc=2)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.title("demo2 figure")
plt.xlabel("label for x")
plt.ylabel("label for y")
# plt.xticks([])
# plt.yticks([])
plt.show()