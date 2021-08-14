import numpy as np
from math import floor, ceil, sqrt
import matplotlib.pyplot as plt

distributions = {'normal': lambda num: np.random.normal(0, 1, num),
                 'cauchy': lambda num: np.random.standard_cauchy(num),
                 'laplace': lambda num: np.random.laplace(0, 1/sqrt(2), num),
                 'poisson': lambda num: np.random.poisson(10, num),
                 'uniform': lambda num: np.random.uniform(-sqrt(3), sqrt(3), num)}


def lab3():
    size_list = [20, 100]

    for dist_name in distributions.keys():
        fig, ax = plt.subplots(1, 1)
        bp_data = [distributions[dist_name](size) for size in size_list]
        trashdata_part = [0] * len(bp_data)
        for _ in range(1000):
            samples = [distributions[dist_name](size) for size in size_list]
            for index, sample in enumerate(samples):
                sample.sort()
                q_1 = sample[floor(1 / 4 * len(sample))]
                q_3 = sample[floor(3 / 4 * len(sample))]
                moustache_left = q_1 - 1.5 * (q_3 - q_1)
                moustache_right = q_3 + 1.5 * (q_3 - q_1)
                trashdata_part[index] +=\
                    (len(sample[sample > moustache_right]) + len(sample[sample < moustache_left])) / len(sample)
        print(f'Distribution name: {dist_name}')
        for idx in range(len(bp_data)):
            print(f'\tSize: {len(bp_data[idx])}, {trashdata_part[idx] / 1000}')

        fig.set(facecolor='lightblue')
        ax.set(facecolor='#E6E6FA')
        ax.set_title(dist_name + ' distribution')
        ax.boxplot(bp_data, vert=False, positions=size_list, widths=[35, 35])
        plt.savefig('../output/' + dist_name + '_boxplot.png', dpi=600)

lab3()
