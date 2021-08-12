import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from math import floor, ceil, sqrt
from scipy.special import factorial


distributions = {'normal': lambda num: np.random.normal(0, 1, num),
                 'cauchy': lambda num: np.random.standard_cauchy(num),
                 'laplace': lambda num: np.random.laplace(0, 1/sqrt(2), num),
                 'poisson': lambda num: np.random.poisson(10, num),
                 'uniform': lambda num: np.random.uniform(-sqrt(3), sqrt(3), num)}


def define_x(name: str, current_array):
    if name == 'poisson':
        new_x = np.array(range(floor(np.min(current_array)) - 1, ceil(np.max(current_array)) + 1, 1))
    else:
        min_ = np.min(current_array)
        max_ = np.max(current_array)
        delta = (max_ - min_) * 0.1
        new_x = np.linspace(min_ - delta, max_ + delta, 500)
    return new_x


def gen_distr_histogram(name: str, pdf, grid, title: str, figName: str):
    distr_10 = distributions.get(name)(10)
    distr_50 = distributions.get(name)(50)
    distr_1000 = distributions.get(name)(1000)

    f = plt.figure(1, figsize=(9, 4))
    plt.subplot(1, 3, 1)
    x = define_x(name, distr_10)
    plt.plot(x, pdf(x), color='maroon')
    plt.hist(distr_10, color='lightblue', density=True, edgecolor='gray')
    plt.title('10 значений')

    plt.subplot(1, 3, 2)
    x = define_x(name, distr_50)
    plt.plot(x, pdf(x), color='maroon')
    plt.hist(distr_50, color='lightblue', density=True, edgecolor='black')
    plt.title('50 значений')

    plt.subplot(1, 3, 3)
    x = define_x(name, distr_1000)
    plt.plot(x, pdf(x), color='maroon')
    plt.hist(distr_1000, color='lightblue', density=True, edgecolor='black')
    plt.title('1000 значений')

    plt.suptitle(title)
    plt.show()
    f.savefig(figName, dpi=200)


def lab1():
    gen_distr_histogram('normal', stats.norm.pdf, (-5, 5), 'Нормальное распределение', '../output/Normal_hist.png')
    gen_distr_histogram('cauchy', stats.cauchy.pdf, (-40, 40), 'Распределение Коши', '../output/Cauchy_hist.png')
    gen_distr_histogram('laplace', stats.laplace.pdf, (-2, 2), 'Распределение Лапласа', '../output/Laplace_hist.png')
    # gen_distr_histogram('poisson', lambda x: np.exp(-10) * np.power(10, x) / factorial(x), (0, 30), 'Распределение Пуассона',
    #                     '../output/Poisson_hist.png')
    gen_distr_histogram('poisson', lambda x: stats.poisson.pmf(x, 10), (0, 30), 'Распределение Пуассона',
                        '../output/Poisson_hist_2.png')
    gen_distr_histogram('uniform', lambda x: stats.uniform.pdf(x, loc=-sqrt(3), scale=2*sqrt(3)), (-8, 8),
                        'Равномерное распределение', '../output/uniform.png')


lab1()
