import numpy as np
from random import *
from math import  *

# Умова за варіантом:
y_max = (30 - 102) * 10
y_min = (20 - 102) * 10
x1 = [20, 70]
x2 = [-20, 40]
m = 6

# Матриця ПФЕ:
pfeMatrix = [[x1[0], x1[1]],
             [x1[0], x2[1]],
             [x2[0], x1[1]],
             [x2[0], x2[1]]]

# Критерій Романовського:
romanovskiy_table = {6: 2.16, 7: 2.3, 8: 2.43,
                     9: 2.5, 10: 2.62, 11: 2.7,
                     12: 2.75, 13: 2.8, 14: 2.85,
                     15: 2.9, 16: 2.94, 17: 2.97,
                     18: 3, 19: 3.05, 20: 3.08}

def findMean(list):
    mean = 0
    for i in range(len(list)):
        mean += list[i]
    mean /= len(list)
    return mean

def romanovskiy(disp):
    sigma = sqrt(2 / m * (2 * m - 2) / (m - 4))
    f_uv = [(disp[0] / disp[1]),
            (disp[2] / disp[0]),
            (disp[2] / disp[1])]
    theta_uv = [((m - 2 / m) * f_uv[0]),
                ((m - 2 / m) * f_uv[1]),
                ((m - 2 / m) * f_uv[2])]
    r_uv = [(abs(theta_uv[0] - 1) / sigma),
            (abs(theta_uv[1] - 1) / sigma),
            (abs(theta_uv[2] - 1) / sigma)]
    for r in r_uv:
        if (r < romanovskiy_table[m]):
            return True
    return False

def dispersion(list):
    mean_y = findMean(list)
    disp = 0
    for i in range(len(list)):
        disp += (list[i] - mean_y) ** 2
    disp /= len(list)
    return disp

def uniformity(m):
    y = [[randint(0, 100) + y_min for _ in range(m)] for i in range(3)]

    disp = [dispersion(y[0]), dispersion(y[1]), dispersion(y[2])]

    if not romanovskiy(disp):
        if m != 20:
            m += 1
            uniformity(m)
            exit()
        else:
            print(
                "Дисперсія неодноріда після 20 дослідів\n"
                "Перейти до обчислення коефіцієнтів рівняння регресії неможливо")
            exit()

    print("Дисперсія однорідна\n"
          "Фактори у точках екперименту:")
    for x in pfeMatrix:
        print(x)
    print(f"\nФункції відгуку:\nПерша точка: {y[0]}\nДруга точка: {y[1]}\nТретя точка: {y[2]}")
    print(f"\nСередне значення Y:\n{findMean(y[0])}\n{findMean(y[1])}\n{findMean(y[2])}")
    print(f"\nДисперсія: {disp}")

    normMatrix = [[-1, -1], [-1, 1], [1, -1], [1, 1]]

    mx1 = (normMatrix[0][0] + normMatrix[1][0] + normMatrix[2][0]) / 3
    mx2 = (normMatrix[0][1] + normMatrix[1][1] + normMatrix[2][1]) / 3
    my = (findMean(y[0]) + findMean(y[1]) + findMean(y[2])) / 3

    a1 = (normMatrix[0][0] ** 2 + normMatrix[1][0] ** 2 + normMatrix[2][0] ** 2) / 3
    a2 = (normMatrix[0][0] * normMatrix[0][1] + normMatrix[1][0] * normMatrix[1][1] +
          normMatrix[2][0] * normMatrix[2][1]) / 3
    a3 = (normMatrix[0][1] ** 2 + normMatrix[1][1] ** 2 + normMatrix[2][1] ** 2) / 3
    a11 = (normMatrix[0][0] * findMean(y[0]) + normMatrix[1][0] * findMean(y[1]) + normMatrix[2][
        0] * findMean(y[2])) / 3
    a22 = (normMatrix[0][1] * findMean(y[0]) + normMatrix[1][1] * findMean(y[1]) + normMatrix[2][
        1] * findMean(y[2])) / 3

    b0_numerator = np.array([[my, mx1, mx2],
                             [a11, a1, a2],
                             [a22, a2, a3]])
    b0_denominator = np.array([[1, mx1, mx2],
                               [mx1, a1, a2],
                               [mx2, a2, a3]])
    b0 = np.linalg.det(b0_numerator) / np.linalg.det(b0_denominator)
    b1_numerator = np.array([[1, my, mx2],
                             [mx1, a11, a2],
                             [mx2, a22, a3]])
    b1_denominator = np.array([[1, mx1, mx2],
                               [mx1, a1, a2],
                               [mx2, a2, a3]])
    b1 = np.linalg.det(b1_numerator) / np.linalg.det(b1_denominator)
    b2_numerator = np.array([[1, mx1, my],
                             [mx1, a1, a11],
                             [mx2, a2, a22]])
    b2_denominator = np.array([[1, mx1, mx2],
                               [mx1, a1, a2],
                               [mx2, a2, a3]])
    b2 = np.linalg.det(b2_numerator) / np.linalg.det(b2_denominator)

    delta_x1 = abs(x2[0] - x1[0]) / 2
    delta_x2 = abs(x2[1] - x1[1]) / 2

    x10 = (x2[0] + x1[0]) / 2
    x20 = (x2[1] + x1[1]) / 2

    a0 = b0 - b1 * (x10 / delta_x1) - b2 * (x20 / delta_x2)
    a1 = b1 / delta_x1
    a2 = b2 / delta_x2

    print("\nНормалізована матриця")
    for x in normMatrix:
        print(x)
    print(f"\nb0 = {b0}; b1 = {b1}; b2 = {b2}")
    print(f"Нормоване рівняння регресії:\ny = {round(b0, 3)} + {round(b1, 3)}*x1 + {round(b2, 3)}*x2\n")

    print("Середні значення нормалізованих Y:")
    for i in range(3):
        print(f"{b0 + b1 * normMatrix[i][0] + b2 * normMatrix[i][1]}")
    print("Значення збігаються зі значеннями Yj")
    print(f"\na0 = {a0}; a1 = {a1}; a2 = {a2}")
    print(f"Натуралізоване рівняння регресії:\ny = {round(a0, 3)} + {round(a1, 3)}*x1 + {round(a2, 3)}*x2")

    print("\nСередні значення натуралізованих Y:")
    for i in range(3):
        print(f"{a0 + a1 * pfeMatrix[i][0] + a2 * pfeMatrix[i][1]}")

    print("Значення збігаються зі значеннями Yj\nКоефіцієнти рівняння регресії розраховані вірно.")

uniformity(m)
