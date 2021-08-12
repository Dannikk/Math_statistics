import numpy as np
from math import floor, ceil, sqrt


distributions = {'normal': lambda num: np.random.normal(0, 1, num),
                 'cauchy': lambda num: np.random.standard_cauchy(num),
                 'laplace': lambda num: np.random.laplace(0, 1/sqrt(2), num),
                 'poisson': lambda num: np.random.poisson(10, num),
                 'uniform': lambda num: np.random.uniform(-sqrt(3), sqrt(3), num)}


def get_analysys(sizes: list, ds: dict):

    for distribution in ds:
        print(f'Name of distribution: {distribution}')
        for size in sizes:
            averages = []
            medians = []
            z_R = []
            z_Q = []
            z_tr = []
            for _ in range(1000):
                sample = ds.get(distribution)(size)
                averages.append(np.mean(sample))

                sample.sort()

                l = floor(size / 2)
                if size % 2:
                    med = sample[l + 1]
                else:
                    med = (sample[l] + sample[l + 1]) / 2
                medians.append(med)

                z_R.append((sample[0] + sample[-1]) / 2)

                x_14 = sample[floor(1 / 4 * size)]
                x_34 = sample[floor(3 / 4 * size)]
                z_Q.append((x_14 + x_34) / 2)

                r = floor(size / 4)
                z_tr_current = 0
                for i in range(r + 1, size - r + 1):
                    z_tr_current += sample[i]
                z_tr.append(z_tr_current / (size - 2*r))

            print(f'Size: {size}')
            print('\taverage and varience of ...')
            print(f'\t\taverages: {np.mean(averages)} and {np.var(averages)}')
            print(f'\t\tmedians: {np.mean(medians)} and {np.var(medians)}')
            print(f'\t\tz_R: {np.mean(z_R)} and {np.var(z_R)}')
            print(f'\t\tz_Q: {np.mean(z_Q)} and {np.var(z_Q)}')
            print(f'\t\tz_tr: {np.mean(z_tr)} and {np.var(z_tr)}')

        print('____________________________________________________')


size_list = [10, 100, 1000]
get_analysys(size_list, distributions)
