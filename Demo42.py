from numpy import array, cov, mean
from numpy.linalg import eig

A = array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])
print(A)
M = mean(A.T, axis=1)
print(M)
M2 = mean(A.T)
print(M2)
M3 = mean(A, axis=1)
print(M3)
C = A - M
print(C)
V = cov(C.T)
print(V)
values, vectors = eig(V)
print("vectors", vectors, sep="\n")
print("values=", values)
P = vectors.T.dot(C.T)
print("project", P.T, sep='\n')
new_vector1 = vectors[:, :2]
P2 = new_vector1.T.dot(C.T)
print('reduce to two vectors', P2.T, sep="\n")
new_vector2 = vectors[:, :1]
P3 = new_vector2.T.dot(C.T)
print("reduce to 1 vectors", P3.T, sep="\n")