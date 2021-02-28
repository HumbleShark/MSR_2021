from random import *
from pprint import *

topBound = 20
matrix = [[randint(1,topBound) for j in range(3)] for i in range(8)]
a0,a1,a2,a3 = [randint(1,topBound) for k in range(4)]
Y = [(a0+a1*matrix[i][0]+a2*matrix[i][1]+a3*matrix[i][2]) for i in range(8)]
x1 = [matrix[i][0] for i in range(8)]
x01 = (max(x1) + min(x1)) / 2
dx1 = x01 - min(x1)
x2 = [matrix[i][1] for i in range(8)]
x02 = (max(x2) + min(x2)) / 2
dx2 = x02 - min(x2)
x3 = [matrix[i][2] for i in range(8)]
x03 = (max(x3) + min(x3)) / 2
dx3 = x03 - min(x3)
x0 = [x01,x02,x03]
dx = [dx1, dx2, dx3]
nMatrix = [[round((matrix[i][j]-x0[j])/dx[j], 3) for j in range(3)] for i in range(8)]

print("Початкова матриця:")
pprint(matrix)
print("-"*32)
print("a0 = {0}, a1 = {1}, a2 = {2}, a3 = {3}.".format(a0,a1,a2,a3))
print("-"*32)
for i in range(8):
    print("Y{0} = ".format(i+1)+ str(Y[i]))
print("-" * 32)
for i in range(3):
    print("x0{0} = ".format(i + 1) + str(x0[i]))
print("-"*32)
for i in range(3):
    print("dx{0} = ".format(i + 1) + str(dx[i]))
print("-" * 32)
print("Нормована матриця:")
pprint(nMatrix)
print("-"*32)
print("Y еталонне: " + str(round((a0+a1*x01+a2*x02+a3*x03), 3)))
print("-"*32)
print("min(Y) = " + str(min(Y)))
print("-"*32)
print("Значення Х, відповідні до min(Y): " + str(matrix[Y.index(min(Y))]))