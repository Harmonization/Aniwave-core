import numpy as np
from scipy.signal import savgol_filter

from bio.parsing import nm

def nadarai(Y, h: int = 3, X = None):
    '''
    Реализация алгоритма Надарая - Ватсона для сглаживания сигнала.
    Скорость фильтрации оптимизирована как для 1 сигнала (спектра ГСИ), так и для целого ГСИ.

    Параметры
    ----------
    Y : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего ГСИ, 
        или 1D массив размера `(n_bands,)` содержащего спектральную сигнатуру.
    h : int
        Ширина окна фильтрации.
    X : np.array | None
       Отвечает за учет данных по оси X, это могут быть номера каналов или нанометры.
    
    Возвращает
    -------
    np.ndarray
        Результирующий спектр или целое изображение ГСИ.
    
    Примеры
    --------
    >>> import numpy as np
    >>> from bio.signal import nadarai
    >>> spectre = np.random.rand(204,)  # имитирует спектр
    >>> result = savgol(spectre, 5)
    >>> print(result.shape)
    (204,)  # Сглаженный спектр
    '''

    # Быстрый алгоритм как для 1 спектра, так и целой hsi
    kernel = lambda x: np.exp(-2*x**2)
    distance = lambda x, y: np.sqrt((x - y)**2)

    if X is None: X = np.array(nm)
    X_T = X[:, np.newaxis]
    w = kernel(distance(X_T, X) / h)
    return Y @ w / np.sum(w, axis=0)

def savgol(Y, h=5):
    '''
    Реализация фильтра Савицкого - Голея для сглаживания спектральных сигнатур у данных ГСИ.
    Скорость фильтрации оптимизирована как для 1 сигнала (спектра ГСИ), так и для целого ГСИ.

    Параметры
    ----------
    Y : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего ГСИ, 
        или 1D массив размера `(n_bands,)` содержащего спектральную сигнатуру.
    h : int
        Ширина окна фильтрации.
    
    Возвращает
    -------
    np.ndarray
        Результирующий спектр или целое изображение ГСИ.
    
    Примеры
    --------
    >>> import numpy as np
    >>> from bio.signal import savgol
    >>> hsi = np.random.rand(100, 100, 20)  # имитирует ГСИ данные
    >>> result = savgol(hsi, 5)
    >>> print(result.shape)
    (100, 100, 20)  # Сглаженное ГСИ
    '''

    if len(Y.shape) == 3:
        return savgol_filter(Y, window_length=h, polyorder=2, axis=2)
    else:
        return savgol_filter(Y, window_length=h, polyorder=2)