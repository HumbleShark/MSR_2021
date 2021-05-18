import math
import numpy as np
from numpy import transpose
from numpy.linalg import solve
from prettytable import PrettyTable
from scipy.stats import f
from scipy.stats import t as t_criterium
from functools import partial
from random import randint


m = 3
N = 8
d = 8

x1_min = -10
x1_max = 50

x2_min = 20
x2_max = 60

x3_min = 50
x3_max = 55

x_max_average = (x1_max + x2_max + x3_max) / 3
x_min_average = (x1_min + x2_min + x3_min) / 3

y_max = int(200 + x_max_average)
y_min = int(200 + x_min_average)

y_matrix = [[randint(y_min, y_max) for _ in range(m)] for _ in range(N)]

average_y = [round(sum(y_matrix[k1]) / m, 3) for k1 in range(N)]

F1 = m - 1
F2 = N
F3 = F1 * F2
F4 = N - d

x1_list = []
x2_list = []
x3_list = []
x1x2_list = []
x1x3_list = []
x2x3_list = []
x1x2x3_list = []

x0_f = [1, 1, 1, 1, 1, 1, 1, 1]
x1_f = [-1, -1, 1, 1, -1, -1, 1, 1]
x2_f = [-1, 1, -1, 1, -1, 1, -1, 1]
x3_f = [-1, 1, 1, -1, 1, -1, -1, 1]

x1x2_f = []
x1x3_f = []
x2x3_f = []
x1x2x3_f = []

for i in range(len(x0_f)):
    x1x2_f.append(x1_f[i] * x2_f[i])
    x1x3_f.append(x1_f[i] * x3_f[i])
    x2x3_f.append(x2_f[i] * x3_f[i])
    x1x2x3_f.append(x1_f[i] * x2_f[i] * x3_f[i])

x_list = [x0_f, x1_list, x2_list, x3_list, x1x2_list, x1x3_list, x2x3_list, x1x2x3_list]
x_f_list = [x0_f, x1_f, x2_f, x3_f, x1x2_f, x1x3_f, x2x3_f, x1x2x3_f]

for i in range(len(x0_f)):
    if x1_f[i] == 1:
        x1_list.append(x1_max)
    else:
        x1_list.append(x1_min)

    if x2_f[i] == 1:
        x2_list.append(x2_max)
    else:
        x2_list.append(x2_min)

    if x3_f[i] == 1:
        x3_list.append(x3_max)
    else:
        x3_list.append(x3_min)

    x1x2_list.append(x1_list[i] * x2_list[i])
    x2x3_list.append(x2_list[i] * x3_list[i])
    x1x3_list.append(x1_list[i] * x3_list[i])

    x1x2x3_list.append(x1_list[i] * x2_list[i] * x3_list[i])


dispersion = [round(sum([((k1 - average_y[j]) ** 2) for k1 in y_matrix[j]]) / m, 3) for j in range(N)]

y_matrix_trans = transpose(y_matrix).tolist()

list_to_solve_1 = list(zip(*x_list))
list_to_solve_2 = x_f_list

list_i_2 = []
for k in range(N):
    S = 0
    for i in range(N):
        S += (list_to_solve_2[k][i] * average_y[i]) / N
    list_i_2.append(round(S, 5))

column_titles = ["X0", "X1", "X2", "X3", "X1X2", "X1X3", "X2X3", "X1X2X3", "Y1", "Y2", "Y3", "Y", "S^2"]

table = PrettyTable()
cols = x_f_list
[cols.extend(ls) for ls in [y_matrix_trans, [average_y], [dispersion]]]
for i in range(13):
    table.add_column(column_titles[i], cols[i])
print(table, "\n")
print('Рівняння регресії з коефіцієнтами нормованих значень:')
print("y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3 \n".format(*list_i_2))

table = PrettyTable()
cols = x_list
[cols.extend(ls) for ls in [y_matrix_trans, [average_y], [dispersion]]]
for i in range(13):
    table.add_column(column_titles[i], cols[i])
print(table, "\n")

list_i_1 = []
for i in solve(list_to_solve_1, average_y):
    list_i_1.append(round(i, 5))
print('Рівняння регресії з коефіцієнтами натуральних значень:')
print("y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3".format(*list_i_1))

Gp = max(dispersion) / sum(dispersion)
q = 0.05
q1 = q / F1
fisher = f.ppf(q=1 - q1, dfn=F2, dfd=(F1 - 1) * F2)
Gt = fisher / (fisher + F1 - 1)

if Gp < Gt:
    print("-" * 100)
    print("Дисперсія однорідна\n")
    dispersion_b = sum(dispersion) / N
    dispersion_beta = dispersion_b / (m * N)
    S_beta = math.sqrt(abs(dispersion_beta))
    beta_list = np.zeros(8).tolist()
    for i in range(N):
        beta_list[0] += (average_y[i] * x0_f[i]) / N
        beta_list[1] += (average_y[i] * x1_f[i]) / N
        beta_list[2] += (average_y[i] * x2_f[i]) / N
        beta_list[3] += (average_y[i] * x3_f[i]) / N
        beta_list[4] += (average_y[i] * x1x2_f[i]) / N
        beta_list[5] += (average_y[i] * x1x3_f[i]) / N
        beta_list[6] += (average_y[i] * x2x3_f[i]) / N
        beta_list[7] += (average_y[i] * x1x2x3_f[i]) / N
    t_list = [abs(beta_list[i]) / S_beta for i in range(0, N)]

    significant = 0
    for i, j in enumerate(t_list):
        if j >= t_criterium.ppf(q=0.975, df=F3):
            print(f'Значний: {beta_list[i]}')
            significant += 1
        else:
            print(f'Незначний: {beta_list[i]}')
            beta_list[i] = 0
            d -= 1
    print("\nКількість значних коефіцієнтів - {}\nКількість незначних коефіцієнтів - {}".format(significant, 8-significant))
    print("-" * 100)
    print("y = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1x2 + {}*x1x3 + {}*x2x3 + {}*x1x2x3".format(*beta_list))
    if significant >= 5:
        print("Кількість значимих коефіцієнтів більша за кількість незначимих, отже система не адекватна")
        exit()
    y_counted = [sum([beta_list[0], *[beta_list[i] * x_list[1:][j][i] for i in range(N)]])
                 for j in range(N)]
    dispersion_ad = 0
    for i in range(len(y_counted)):
        dispersion_ad += ((y_counted[i] - average_y[i]) ** 2) * m / (N - d)
    Fp = dispersion_ad / dispersion_beta
    fisher = partial(f.ppf, q=1 - 0.05)
    Ft = fisher(dfn=F4, dfd=F3)

    if Ft > Fp:
        print("Рівняння регресії адекватне")
    else:
        print("Рівняння регресії неадекватне")
else:
    print("Дисперсія неоднорідна!")