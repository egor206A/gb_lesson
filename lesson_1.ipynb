# Итак, будем работать с датасетом Digits из Sklearn, так как его можно довольно просто получить в две строчки кода:
# импортируем библиотеки
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

# собственно, загрузка датасета
x, y = load_digits(n_class=10, return_X_y=True)

print(x.shape)
>>> (1797, 64)
print(np.unique(y))
>>> array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

index = 0

print(y[index])
plt.imshow(x[index].reshape(8, 8), cmap="Greys")

x[index]
>>> array([ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.,  0.,  0., 13., 15., 10.,
       15.,  5.,  0.,  0.,  3., 15.,  2.,  0., 11.,  8.,  0.,  0.,  4.,
       12.,  0.,  0.,  8.,  8.,  0.,  0.,  5.,  8.,  0.,  0.,  9.,  8.,
        0.,  0.,  4., 11.,  0.,  1., 12.,  7.,  0.,  0.,  2., 14.,  5.,
       10., 12.,  0.,  0.,  0.,  0.,  6., 13., 10.,  0.,  0.,  0.])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# разделим выборку на тренировочную и тестовую
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    shuffle=True, random_state=42)
print(x_train.shape)
>>> (1437, 64)
print(x_test.shape)
>>> (360, 64)

scaler = StandardScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
