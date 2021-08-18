import numpy as np
import scipy.stats as stats
from math import floor, ceil, sqrt, exp
import matplotlib.pyplot as plt

distributions = {'normal': lambda num: np.random.normal(0, 1, num),
                 'cauchy': lambda num: np.random.standard_cauchy(num),
                 'laplace': lambda num: np.random.laplace(0, 1/sqrt(2), num),
                 'poisson': lambda num: np.random.poisson(10, num),
                 'uniform': lambda num: np.random.uniform(-sqrt(3), sqrt(3), num)}

pdfs = {'normal': stats.norm.pdf,
        'cauchy': stats.cauchy.pdf,
        'laplace': stats.laplace.pdf,
        'poisson': lambda x: stats.poisson.pmf(x, 10),
        'uniform': lambda x: stats.uniform.pdf(x, loc=-sqrt(3), scale=2*sqrt(3))}

cdfs = {
    'normal': lambda x: stats.norm.cdf(x, 0, 1),
    'cauchy': lambda x: stats.cauchy.cdf(x, 0, 1),
    'laplace': lambda x: stats.laplace.cdf(x, 0, sqrt(2) / 2),
    'poisson': lambda x: stats.poisson.cdf(x, 10),
    'uniform': lambda x: stats.uniform.cdf(x, -sqrt(3), 2 * sqrt(3))}


size_list = [20, 60, 100]
h_coefs = [0.5, 1, 2]


def define_x(name: str, current_array):
    min_ = np.min(current_array)
    max_ = np.max(current_array)
    delta = (max_ - min_) * 0.1
    new_x = np.linspace(min_ - delta, max_ + delta, 500)
    return new_x


def K(u: float):
    return 1/sqrt(2*np.pi)*np.exp(-u**2 / 2)


def get_h_n(sample: np.ndarray):
    return 1.06 * np.std(sample) * (len(sample) ** (-1 / 5))


def f_assessment(x: np.ndarray, sample: np.ndarray, h_coef: float):
    f_arr = []
    h_n = get_h_n(sample)
    for i in x:
        f_arr.append(1 / len(sample) / (h_n * h_coef) * sum([K((i - x_i) / (h_n * h_coef)) for x_i in sample]))
    return np.array(f_arr)


def kernel_assessments(size_list: list):
    for distr_name in distributions.keys():
        for size in size_list:
            if distr_name == 'poisson':
                x = np.linspace(6, 14, 100)
                x_pdf = np.array(range(6, 15, 1))
            else:
                x = np.linspace(-4, 4, 100)
                x_pdf = np.linspace(-4, 4, 100)

            f = plt.figure(1, figsize=(9, 4))
            for index, h_coef in enumerate(h_coefs):
                plt.subplot(1, 3, index + 1)

                sample = distributions[distr_name](size)
                y = f_assessment(x, sample, h_coef)
                plt.grid()
                plt.plot(x, y, color='red')
                plt.plot(x_pdf, pdfs[distr_name](x_pdf), color='blue')
                plt.title('h*' + str(h_coef))
                if index == len(h_coefs) - 1:
                    plt.legend(['kernel assessment', 'density function'])

            plt.suptitle(distr_name + ' kernel ' + str(size))
            plt.show()
            f.savefig('../output/4_lab/' + distr_name + '_kernel_' + str(size), dpi=200)


def emp_cdf(x: np.ndarray, sample: np.ndarray):
    f_arr = []
    for i in x:
        freq = (len(sample[sample < i])) / len(sample)
        f_arr.append(freq)
    return np.array(f_arr)


def empirical_distr_function_analysys(size_list: list):
    for distr_name in distributions.keys():
        f = plt.figure(1, figsize=(9, 4))
        for index, size in enumerate(size_list):
            plt.subplot(1, 3, index + 1)

            sample = distributions[distr_name](size)
            x = define_x(distr_name, sample)
            plt.plot(x, emp_cdf(x, sample), color='red')
            plt.plot(x, cdfs[distr_name](x), color='blue')
            plt.title('Size: ' + str(size))
            if index == len(h_coefs) - 1:
                plt.legend(['empirical cdf', 'theoretical cdf'])

        plt.suptitle(distr_name + ' distribution: empirical and theoretical distribution functions')
        plt.show()
        f.savefig('../output/4_lab/' + distr_name + '_cdfs', dpi=200)


# kernel_assessments(size_list)
empirical_distr_function_analysys(size_list)