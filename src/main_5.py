import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize, fsolve
from math import sqrt
from prettytable import PrettyTable as Ptable
import matplotlib.pyplot as plt

size_list = [20, 60, 100]
rho_list = [0, 0.5, 0.9]
distr_params = {'mx': 0,
                'my': 0,
                'stdx': 1,
                'stdy': 1}

distr_params_1 = {'mx': 0,
                'my': 0,
                'stdx': 1,
                'stdy': 1,
                'rho': 0.9,
                'coeff': 0.9}

distr_params_2 = {'mx': 0,
                  'my': 0,
                  'stdx': 10,
                  'stdy': 10,
                  'rho': -0.9,
                  'coeff': 0.1}


def get_normal_2d_sample(mx: float, my: float, stdx: float, stdy: float, rho: float, size: int) -> np.ndarray:
    mean = [mx, my]
    covxy = rho * stdx * stdy
    cov = [[stdx**2, covxy],
           [covxy, stdy**2]]
    return np.random.multivariate_normal(mean=mean, cov=cov, size=size)


def get_params(sample: np.ndarray) -> dict:
    result = dict()
    mean_x = np.mean(sample[:, 0])
    mean_y = np.mean(sample[:, 1])
    result['r_p'] = sum((sample[:, 0] - mean_x) * (sample[:, 1] - mean_y)) / \
                    sqrt(sum((sample[:, 0] - mean_x) ** 2) * sum((sample[:, 1] - mean_y) ** 2))

    result['r_s'], _ = stats.spearmanr(sample)

    n13, n24 = 0, 0
    med_x, med_y = tuple(np.median(sample, axis=0))
    for point in sample:
        if (point[0] - med_x) * (point[1] - med_y) >= 0:
            n13 += 1
        else:
            n24 += 1
    result['r_q'] = (n13 - n24) / len(sample)

    return result


def normal_2d_analysys(size_list: list, analysys_size: int, rho_list: list, params: dict):
    for size in size_list:
        print(f'Размер выборки: {size}')
        table = Ptable()
        table.field_names = ['Показатель', 'rho', 'r_p', 'r_s', 'r_q']
        for rho in rho_list:
            r_p_l, r_s_l, r_q_l = [], [], []
            for _ in range(analysys_size):
                sample = get_normal_2d_sample(mx=params['mx'], my=params['my'],
                                              stdx=params['stdx'], stdy=params['stdy'],
                                              rho=rho, size=size)
                res_params = get_params(sample)
                r_p_l.append(res_params['r_p'])
                r_s_l.append(res_params['r_s'])
                r_q_l.append(res_params['r_q'])

            r_p_l = np.array(r_p_l)
            r_s_l = np.array(r_s_l)
            r_q_l = np.array(r_q_l)

            table.add_row(['E(z)', rho, round(np.mean(r_p_l), 3), round(np.mean(r_s_l), 3), round(np.mean(r_q_l), 3)])
            table.add_row(['E(z^2)', rho, round(np.mean(r_p_l**2), 3), round(np.mean(r_s_l**2), 3), round(np.mean(r_q_l**2), 3)])
            table.add_row(['D(z)', rho, round(np.var(r_p_l), 3), round(np.var(r_s_l), 3), round(np.var(r_q_l), 3)])
            table.add_row(['', '', '', '', ''])

        print(table)


def mix_normal_2d_analysys(size_list: list, analysys_size: int, params_1: dict, params_2: dict):
    table = Ptable()
    table.field_names = ['Показатель', 'размер', 'r_p', 'r_s', 'r_q']
    print('Исследование смешанного распределения')
    for size in size_list:
        r_p_l, r_s_l, r_q_l = [], [], []
        for _ in range(analysys_size):
            sample_1 = get_normal_2d_sample(mx=params_1['mx'], my=params_1['my'],
                                            stdx=params_1['stdx'], stdy=params_1['stdy'],
                                            rho=params_1['rho'], size=size)
            sample_2 = get_normal_2d_sample(mx=params_2['mx'], my=params_2['my'],
                                            stdx=params_2['stdx'], stdy=params_2['stdy'],
                                            rho=params_2['rho'], size=size)
            sample = sample_1 * params_1['coeff'] + sample_2 * params_2['coeff']

            res_params = get_params(sample)
            r_p_l.append(res_params['r_p'])
            r_s_l.append(res_params['r_s'])
            r_q_l.append(res_params['r_q'])

        r_p_l = np.array(r_p_l)
        r_s_l = np.array(r_s_l)
        r_q_l = np.array(r_q_l)

        table.add_row(['E(z)', size, round(np.mean(r_p_l), 3), round(np.mean(r_s_l), 3), round(np.mean(r_q_l), 3)])
        table.add_row(['E(z^2)', size, round(np.mean(r_p_l ** 2), 3), round(np.mean(r_s_l ** 2), 3),
                       round(np.mean(r_q_l ** 2), 3)])
        table.add_row(['D(z)', size, round(np.var(r_p_l), 3), round(np.var(r_s_l), 3), round(np.var(r_q_l), 3)])
        table.add_row(['', '', '', '', ''])
    print(table)


def y(x, mx: float, my: float, stdx: float, stdy: float, rho: float, rad_2: float):
    y_1 = stdy * (rho * (x - mx) / stdx + np.sqrt((x - mx) ** 2 * (rho ** 2 - 1) / (stdx ** 2) + rad_2)) + my
    y_2 = stdy * (rho * (x - mx) / stdx - np.sqrt((x - mx) ** 2 * (rho ** 2 - 1) / (stdx ** 2) + rad_2)) + my
    return y_1, y_2


# def temp_fun(x):
#     return y(x, 0, 0, 1, 1, 0.8, 1)


def under_sqrt_fun(x, mx: float, my: float, stdx: float, stdy: float, rho: float, rad_2: float):
    res = (x - mx) ** 2 * (rho ** 2 - 1) / (stdx ** 2) + rad_2
    return res


def define_x_interval(mx: float, my: float, stdx: float, stdy: float, rho: float, rad_2: float):
    fun = lambda x: under_sqrt_fun(x, mx=mx, my=my, stdx=stdx, stdy=stdy, rho=rho, rad_2=rad_2)
    roots = fsolve(fun, [-10, 10])
    roots.sort()
    return np.linspace(roots[0], roots[1] + (roots[1] - roots[0]) * 0.001, 1000)


def ellipses_plotting(size_list: list, rho_list: list, params: dict, rad_2: float):
    for size in size_list:
        f = plt.figure(1, figsize=(9, 4))
        for index, rho in enumerate(rho_list):
            plt.subplot(1, 3, index + 1)
            sample = get_normal_2d_sample(mx=params['mx'], my=params['my'],
                                          stdx=params['stdx'], stdy=params['stdy'],
                                          rho=rho, size=size)
            x = define_x_interval(mx=params['mx'], my=params['my'], stdx=params['stdx'], stdy=params['stdy'],
                                  rho=rho, rad_2=rad_2)
            y1, y2 = y(x, mx=params['mx'], my=params['my'], stdx=params['stdx'], stdy=params['stdy'],
                       rho=rho, rad_2=rad_2)
            plt.plot(x, y1, color='blue')
            plt.plot(x, y2, color='blue')
            plt.scatter(sample[:, 0], sample[:, 1], color='red', s=3)
            plt.grid(True)
            plt.title(f'Corr = {rho}')

        plt.suptitle(f'Sample size: {size}')
        plt.show()
        f.savefig('../output/5_lab/ellipse' + str(size), dpi=200)


normal_2d_analysys(size_list=size_list, analysys_size=1000, rho_list=rho_list, params=distr_params)
mix_normal_2d_analysys(size_list=size_list, analysys_size=1000, params_1=distr_params_1, params_2=distr_params_2)
ellipses_plotting(size_list, rho_list, distr_params, 4)