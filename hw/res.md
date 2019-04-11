

```python
import numpy as np
import pandas as pd
```

# 1


```python
x = np.array([5, 2, np.nan, 2.3, 6, 7, np.nan, 8.5])
s = pd.Series(x)
x = x[~np.isnan(x)]
x
```




    array([ 5. ,  2. ,  2.3,  6. ,  7. ,  8.5])




```python
# numpy.r_[True, a[1:] < a[:-1]] & numpy.r_[a[:-1] < a[1:], True]
```


```python
s.dropna()
```




    0    5.0
    1    2.0
    3    2.3
    4    6.0
    5    7.0
    7    8.5
    dtype: float64



# 2


```python
from scipy.signal import argrelextrema

x = np.array([1, 20, 12, 4, 22, 10, 11])
x[argrelextrema(x, np.greater)[0]]
```




    array([20, 22])



# 3


```python
a = np.arange(1000)
mask = np.arange(901)[:, None] + np.arange(100)[None, :]
a[mask]
```




    array([[  0,   1,   2, ...,  97,  98,  99],
           [  1,   2,   3, ...,  98,  99, 100],
           [  2,   3,   4, ...,  99, 100, 101],
           ..., 
           [898, 899, 900, ..., 995, 996, 997],
           [899, 900, 901, ..., 996, 997, 998],
           [900, 901, 902, ..., 997, 998, 999]])



# 5


```python
from itertools import permutations
```


```python
class RDATA(object):

    def __init__(self, max_samples = 64, n = 10 ** 7):
        self.n = n
        self._max_samples = max_samples

        eps = 1 + np.random.normal(0, 0.01, n)
        eps[0] = 10
        
        self.ID = np.cumprod(eps)
        self.DataS = np.zeros(self.ID.shape[0])
        self.DataS[1:] = np.diff(self.ID)
    
    def _split4_data(self):
        'Разбиваем на 4ки'
        
        buf_data = self.batch.reshape((64, 4, 128))

        perms = np.array(list(permutations([0, 1, 2, 3])))
        Y = np.random.randint(0, 24, 64)
        
        'Индекс-маска переставляет 4ки между собой'

        mask = perms[Y] + 4 * np.repeat(np.arange(64), 4).reshape((64, 4))
        return buf_data.reshape((64 * 4, 128))[mask]

    def loader(self, batch_size=64):
        """
        Гератор новых батчей
        batch_size - размер батча
        
        Генератор следовало бы реализовывать несколько по-другому,
        но задание есть задание
        """

        n = self._max_samples
        ind = np.arange(self._max_samples)

        "В mask выбирается 64 семпла длиной 512"

        mask = np.random.choice(
            np.arange(self.DataS.shape[0] - 512), 
            size=64)[:, None] + np.arange(512)[None, :]

        """
        batch - массив, который требуется в задании
        Можно сразу же возвращать его (он будет размера (64, 512))
        Цикл здесь требуется только для самой "генерации",
        хоть сам batch сделан без циклов
        """
        
        self.batch = self.DataS[mask]
        #разбивает не 4ки и переставляет их
        self.batch = self._split4_data().reshape((64, 512))
        
        for i in range(0, n, batch_size):
            mask = ind[i: min(i + batch_size, n)]
            yield (self.batch[mask] if mask.shape[0] != 1 else self.batch[mask][0])
        
    def get_data(self):
        return next(iter(self.loader()))
```


```python
# ex

data = RDATA()
X = data.get_data()
```

# Задания Данила

## 3


```python
from sklearn.preprocessing import MinMaxScaler
```


```python
sc = MinMaxScaler()
data = np.random.randint(0, 20, size=(10, 5))
sc.fit_transform(data.T).T
```

    /Users/kosov/anaconda/lib/python3.6/site-packages/sklearn/utils/validation.py:590: DataConversionWarning: Data with input dtype int64 was converted to float64 by MinMaxScaler.
      warnings.warn(msg, DataConversionWarning)





    array([[ 0.41176471,  0.88235294,  0.        ,  0.64705882,  1.        ],
           [ 1.        ,  0.        ,  0.35714286,  0.14285714,  0.14285714],
           [ 0.57894737,  1.        ,  0.05263158,  0.        ,  0.94736842],
           [ 0.        ,  0.64705882,  0.47058824,  1.        ,  0.94117647],
           [ 0.        ,  0.68421053,  1.        ,  0.84210526,  0.        ],
           [ 0.875     ,  0.        ,  0.625     ,  0.75      ,  1.        ],
           [ 0.91666667,  1.        ,  0.33333333,  0.83333333,  0.        ],
           [ 0.25      ,  0.58333333,  0.        ,  0.58333333,  1.        ],
           [ 0.        ,  0.4375    ,  0.375     ,  0.4375    ,  1.        ],
           [ 0.77777778,  1.        ,  0.22222222,  0.44444444,  0.        ]])




```python
((data.T - data.min(axis=1)) / (data.max(axis=1) - data.min(axis=1))).T
```




    array([[ 0.41176471,  0.88235294,  0.        ,  0.64705882,  1.        ],
           [ 1.        ,  0.        ,  0.35714286,  0.14285714,  0.14285714],
           [ 0.57894737,  1.        ,  0.05263158,  0.        ,  0.94736842],
           [ 0.        ,  0.64705882,  0.47058824,  1.        ,  0.94117647],
           [ 0.        ,  0.68421053,  1.        ,  0.84210526,  0.        ],
           [ 0.875     ,  0.        ,  0.625     ,  0.75      ,  1.        ],
           [ 0.91666667,  1.        ,  0.33333333,  0.83333333,  0.        ],
           [ 0.25      ,  0.58333333,  0.        ,  0.58333333,  1.        ],
           [ 0.        ,  0.4375    ,  0.375     ,  0.4375    ,  1.        ],
           [ 0.77777778,  1.        ,  0.22222222,  0.44444444,  0.        ]])



## 4


```python
data = np.random.normal(0, 1, size=(100, 5))
```


```python
np.where((data < 0).sum(axis = 1) > 2)[0]
```




    array([ 5,  6,  7,  8,  9, 10, 12, 14, 15, 17, 19, 20, 22, 25, 26, 30, 33,
           34, 36, 40, 42, 43, 46, 47, 48, 50, 51, 53, 55, 56, 57, 58, 59, 62,
           64, 65, 66, 67, 71, 74, 75, 76, 77, 78, 79, 81, 83, 85, 86, 87, 88,
           94, 96, 97, 98, 99])



## 5


```python
A = np.random.randint(0, 4, size=(10, 4))
B = np.random.randint(0, 4, size=(100, 4))
```


```python
A[(A[:, None, :] == B).all(axis=2).any(axis=1)]
```




    array([[2, 3, 2, 2],
           [2, 0, 1, 3],
           [3, 0, 1, 2],
           [3, 2, 2, 0],
           [3, 1, 2, 1]])




```python
A = np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
B = np.array([[1, 2, 3], [1, 1, 1], [1, 2, 3], [6, 7, 8], [1, 2, 3], [1, 1, 1]])
```


```python
A[(A[:, None, :] == B).all(axis=2).any(axis=1)]
```




    array([[1, 2, 3],
           [6, 7, 8]])



## 6


```python
A = np.arange(120).reshape((2, 3, 4, 5))
```


```python
A.sum(axis=-1).sum(axis=-1)
```




    array([[ 190,  590,  990],
           [1390, 1790, 2190]])




```python
A.reshape((2, 3, 20)).sum(axis=-1)
```




    array([[ 190,  590,  990],
           [1390, 1790, 2190]])


