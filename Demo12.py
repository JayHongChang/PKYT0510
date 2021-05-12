import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a.view()
c = a
print(a, '\n', b, '\n', c)
b.shape = (4, 1)
print("after b change shape")
print(a, '\n', b, '\n', c)
print("after c change shape")
c.shape = (1, 4)
print(a, '\n', b, '\n', c)
a[0][0] = 100
print("after a change to 100")
print(a, '\n', b, '\n', c)