import random
import sklearn.linear_model as lm
from scipy.stats import f, t
from functools import partial
from pyDOE2 import *
import pandas as pd
from tabulate import tabulate
from scipy.stats import f


def add_sq(x):
    for i in range(len(x)):
        x[i][4] = x[i][1] * x[i][2]
        x[i][5] = x[i][1] * x[i][3]
        x[i][6] = x[i][2] * x[i][3]
        x[i][7] = x[i][1] * x[i][3] * x[i][2]
        x[i][8] = x[i][1] ** 2
        x[i][9] = x[i][2] ** 2
        x[i][10] = x[i][3] ** 2
    return x


def regression(x, b):
    return sum([x[i] * b[i] for i in range(len(x))])


def criteria_studenta(x, y_aver):
    S_kv = s_kv(y_aver)
    s_kv_aver = sum(S_kv) / n

    s_Bs = (s_kv_aver / n / m) ** 0.5
    Bs = bs(x, y_aver)
    ts = [round(abs(B) / s_Bs, 3) for B in Bs]

    return ts


def bs(x, y_aver):
    res = [sum(y_aver) / n]

    for i in range(len(x[0])):
        b = sum(j[0] * j[1] for j in zip(x[:, i], y_aver)) / n
        res.append(b)

    return res


def s_kv(y_aver):
    res = []
    for i in range(n):
        s = sum([(y_aver[i] - y[i][j]) ** 2 for j in range(m)]) / m
        res.append(round(s, 3))
    return res


def experiment(n, m):
    # Створення матриці

    for i in range(n):
        for j in range(m):
            y[i][j] = random.randint(y_min, y_max)
    if n > 14:
        no = n - 14
    else:
        no = 1

    x_norm = ccdesign(3, center=(0, no))
    x_norm = np.insert(x_norm, 0, 1, axis=1)

    for i in range(4, 11):
        x_norm = np.insert(x_norm, i, 0, axis=1)

    l = 1.215

    for i in range(len(x_norm)):
        for j in range(len(x_norm[i])):
            if x_norm[i][j] < -1 or x_norm[i][j] > 1:
                if x_norm[i][j] < 0:
                    x_norm[i][j] = -l
                else:
                    x_norm[i][j] = l

    x_norm = add_sq(x_norm)

    x = np.ones(shape=(len(x_norm), len(x_norm[0])), dtype=np.int64)

    x_range = [[x1_min, x1_max], [x2_min, x2_max], [x3_min, x3_max]]
    for i in range(8):
        for j in range(1, 4):
            if x_norm[i][j] == -1:
                x[i][j] = x_range[j - 1][0]
            else:
                x[i][j] = x_range[j - 1][1]

    for i in range(8, len(x)):
        for j in range(1, 3):
            x[i][j] = (x_range[j - 1][0] + x_range[j - 1][1]) / 2

    dx = [x_range[i][1] - (x_range[i][0] + x_range[i][1]) / 2 for i in range(3)]

    x[8][1] = l * dx[0] + x[9][1]
    x[9][1] = -l * dx[0] + x[9][1]
    x[10][2] = l * dx[1] + x[9][2]
    x[11][2] = -l * dx[1] + x[9][2]
    x[12][3] = l * dx[2] + x[9][3]
    x[13][3] = -l * dx[2] + x[9][3]
    x = add_sq(x)

    show_arr = pd.DataFrame(x)
    print('\nМатриця X:\n', tabulate(show_arr, headers='keys', tablefmt='psql'))

    show_arr = pd.DataFrame(x_norm)
    print('\nНормована матриця X:\n', tabulate(show_arr.round(0), headers='keys', tablefmt='psql'))

    show_arr = pd.DataFrame(y)
    print('\nМатриця Y:\n', tabulate(show_arr, headers='keys', tablefmt='psql'))

    y_average = [round(sum(i) / len(i), 3) for i in y]

    # Знайдемо коефіціенти

    skm = lm.LinearRegression(fit_intercept=False)
    skm.fit(x, y_average)
    b = skm.coef_

    print('\nКоефіцієнти рівняння регресії:')
    b = [round(i, 3) for i in b]
    print("y = {} +{}*x1 +{}*x2 +{}*x3 + {}*x1*x2 + {}*x1*x3 + {}*x2*x3 + b{}*x1*x2*x3 + {}x1^2 + {}x2^2 + {}x3^2\n".format(*b))
    print('\nРезультат рівняння зі знайденими коефіцієнтами:')
    print(np.dot(x, b))
    print("-" * 100)

    # Проведемо перевірку

    print('\nПеревірка рівняння:')
    f1 = m - 1
    f2 = n
    f3 = f1 * f2
    q = 0.05
    q1 = q / f1

    student = partial(t.ppf, q=1 - q)
    t_student = student(df=f3)

    fisher_value = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
    G_kr = fisher_value / (fisher_value + f1 - 1)

    y_average = [round(sum(i) / len(i), 3) for i in y]
    print('\nСереднє значення y:', y_average)

    disp = []
    for i in range(n):
        s = sum([(y_average[i] - y[i][j]) ** 2 for j in range(m)]) / m
        disp.append(round(s, 3))
    print('Дисперсія y:', disp)

    Gp = max(disp) / sum(disp)
    print('\nПеревіримо за критерієм Кохрена:')

    print(f'Gp = {Gp}')
    if Gp < G_kr:
        print(f'З ймовірністю {1 - q}')
    else:
        print("Збільшимо кількість дослідів")
        m += 1
        experiment(n, m)

    ts = criteria_studenta(x_norm[:, 1:], y_average)
    print('\nКритерій Стьюдента:')
    print(ts)
    res = [t for t in ts if t > t_student]
    final_k = [b[i] for i in range(len(ts)) if ts[i] in res]
    print('\nКоефіцієнти, які не мають статистичного значення:')
    print([round(i, 3) for i in b if i not in final_k])

    y_new = []
    for j in range(n):
        y_new.append(round(regression([x[j][i] for i in range(len(ts)) if ts[i] in res], final_k), 3))

    print(f'\nЗначення y з коефіцієнтами: {final_k}')
    print(y_new)

    d = len(res)
    if d >= n:
        print('\nF4 <= 0')

    f4 = n - d

    s_ad = m / (n - d) * sum([(y_new[i] - y_average[i]) ** 2 for i in range(len(y))])
    s_kv_aver = sum(disp) / n
    f_p = s_ad / s_kv_aver
    fisher = partial(f.ppf, q=0.95)
    f_t = fisher(dfn=f4, dfd=f3)
    print("-" * 100)
    print('\nПеревірка адекватності за критерієм Фішера')
    print('Fp =', f_p)
    print('F_t =', f_t)
    if f_p < f_t:
        print('Математична модель адекватна')
    else:
        print('Математична модель неадекватна!')


n = 15
m = 6

x1_min = -8
x1_max = 9

x2_min = -6
x2_max = 2

x3_min = -1
x3_max = 5

x_average_max = (x1_max + x2_max + x3_max) / 3
x_average_min = (x1_min + x2_min + x3_min) / 3

y_max = 200 + int(x_average_max)
y_min = 200 + int(x_average_min)

y = np.zeros(shape=(n, m))

experiment(n, m)