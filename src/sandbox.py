from math import sqrt
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pyautogui as pag
from time import sleep
from pynput.keyboard import Key, Listener


# m = float(input('Enter mean and varience'))
# var = float(input())
# m = float(m)
# var = float(var)
# print(f'{m} - sqrt({var}) = {m - sqrt(var)}')
# print(f'{m} + sqrt({var}) = {m + sqrt(var)}')

# l = np.array(range(10))
# print(l[l > 5])
# print(sum(l))
# foo1 = lambda x: stats.poisson.pmf(x, 10)
# foo = lambda x: stats.poisson.cdf(x, 10)
# x = np.linspace(5, 20, 1000)
# x1 = np.array(range(2, 20, 1))
# y1 = foo1(x1)
# plt.plot(x, foo(x))
# plt.show()

# class Keylogger:
#
#     def __init__(self):
#         self.count = 0
#         self.keys = []
#
#     def on_press(self, key):
#         print(f'{key} pressed')
#         self.keys.append(key)
#
#     def on_release(self, key):
#         if key == Key.esc:
#             return False
#
# obj = Keylogger()
# with Listener(on_press=obj.on_press, on_release=obj.on_release) as listener:
#     listener.join()

l = np.linspace(-1.8, 2, 20)
l2 = np.linspace(-2, -1, 20)
print(l)
print(l2)
print(np.sign(l))
print(np.sign(l2))
print(np.sign(l)*np.sign(l2))