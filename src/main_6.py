import numpy as np
import matplotlib.pyplot as plt
from math import floor


b_params = (2, 2)
params = {'line_segment': (-1.8, 2),
          'size': 20}
deviations = {'without_deviations': (0, 0),
              'with_deviations': (10, -10)}


def get_l_j(size: int) -> tuple:
    if size % 4 == 0:
        l = size / 4 - 1
    else:
        l = floor(size / 4)
    j = size - l - 1
    return int(l), int(j)


def get_errors(size: int) -> np.array:
    errs = np.random.normal(0, 1, size)
    return errs


def get_y(x: np.array, b_params: tuple):
    return b_params[0] + b_params[1] * x + get_errors(len(x))


def lsm(x: np.array, y: np.array) -> tuple:
    """
    least squares method

    Parameters
    ----------
    x :
    y :
    """
    mx = np.mean(x)
    mx2 = np.mean(x ** 2)
    my = np.mean(y)
    mxy = np.mean(x * y)
    b_1 = (mxy - mx * my) / (mx2 - mx ** 2)
    b_0 = my - mx * b_1
    return b_0, b_1


def lad(x: np.array, y: np.array) -> tuple:
    """
    least absolute deviations

    Parameters
    ----------
    x :
    y :
    """
    med_x = np.median(x, axis=0)
    med_y = np.median(y, axis=0)

    r_Q = np.mean(np.sign(x - med_x) * np.sign(y - med_y))
    l, j = get_l_j(len(x))
    q_y = y[j] - y[l]
    q_x = x[j] - x[l]

    b_1 = r_Q * q_y / q_x
    b_0 = med_y - b_1 * med_x

    return b_0, b_1


def regression_analisys(b_params: tuple, deviations: dict, params: dict):
    x_points = np.linspace(params['line_segment'][0], params['line_segment'][1], params['size'])
    x = np.linspace(params['line_segment'][0], params['line_segment'][1], 100)
    y = b_params[0] + b_params[1] * x
    for deviation in deviations:
        f = plt.figure(1, figsize=(9, 4))
        y_points = get_y(x_points, b_params=b_params)

        y_points[0] += deviations[deviation][0]
        y_points[-1] += deviations[deviation][1]

        b_lsm = lsm(x_points, y_points)
        b_lad = lad(x_points, y_points)
        print('Коэффициенты линейной регрессии: ' + deviation)
        print(f'\tМНК: a = {b_lsm[0]}, b = {b_lsm[1]}')
        print(f'\tМНМ: a = {b_lad[0]}, b = {b_lad[1]}')

        y_lsm = b_lsm[0] + b_lsm[1] * x
        y_lad = b_lad[0] + b_lad[1] * x

        plt.grid(True)
        plt.scatter(x_points, y_points, color='blue')
        print(x_points.shape)
        print(type(x_points))
        plt.plot(x, y, color='red')
        plt.plot(x, y_lsm, color='green')
        plt.plot(x, y_lad, color='purple')

        plt.legend(['reference model', 'LSM (МНК)', 'LAD (МНМ)'])
        plt.title(deviation)
        plt.show()
        f.savefig('../output/6_lab/regression_analysys_' + deviation, dpi=200)


regression_analisys(b_params=b_params, deviations=deviations, params=params)