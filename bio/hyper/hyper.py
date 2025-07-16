import numpy as np
import spectral.io.envi as envi

from bio.signal import savgol, nadarai
from bio.parsing import str2indx
from bio.statistics import sign_info, band_info

# Методы фильтрации спектральных сигнатур
hsi_filters = {'golay': savgol, 'nadarai': nadarai}

# def open_hsi(name: str):
#     # Открыть HSI
    
#     filename, filetype = name.split('.')
#     if filetype == 'hdr':
#         path = f'Data/{filename}.hdr'
#         img = envi.open(path)
#         arr = np.array(img.load())
#         hsi = np.rot90(arr, k=1)
    
#     elif filetype == 'npy':
#         path = f'Data/{filename}.npy'
#         hsi = np.load(path)

#     return hsi

def rgb(hsi: np.ndarray, bands: tuple = (70, 51, 18)):
    '''
    Синтез псевдо-RGB изображения для ГСИ растений снятых со спектрометра Specim IQ. 
    Красный зеленый и синий каналы заранее определены. 
    Изображение автоматически переворачивается для соответствия оригинальному со стороны фронтенда.

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего гиперспектральный данные.
    
    Возвращает
    -------
    np.ndarray
        3D массив numpy размера (n_rows, n_columns, 3) содержащий синтезированное RGB изображение для визуализации.
    
    Примеры
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from bio.hyper import rgb
    >>> hsi = np.random.rand(100, 100, 20)  # имитирует ГСИ данные
    >>> result = rgb(hsi)
    >>> plt.imshow(result)
    >>> plt.show()
    >>> print(result.shape)
    (100, 100, 3)  # Example output
    '''
    return np.flip(np.around(hsi[..., bands] * 255).astype(int), axis=0)

def signal(hsi: np.ndarray, i: int, j: int, method: str = '', h: int = 5):
    '''
    Получение спектральной сигнатуры HSI по указанным координатам пикселей.
    Функция поддерживает сглаживание сигнатуры одним из методов фильтрации/сглаживания.
    Выполняет роль не только геттера, но и выполняет статистический анализ полученного спектра,
    получая его статистические признаки, производную, и статистические признаки производной.
    Сглаживание сигнатуры рекомендуется для нахождения более плавных производных.

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего гиперспектральный данные.
    i : int
        Номер ряда в ГСИ у нужного пикселя.
    j : int
        Номер столбца в ГСИ у нужного пикселя.
    method : str
        Метод сглаживания спектральной сигнатуры. При указании пустого значения фильтрация не производится.
    h : int
        Ширина окна сглаживания, является параметром методов фильтрации. Чем больше окно тем сильнее сглаживание.
    
    Возвращает
    -------
    dict
        Словарь, содержащий 1D массив numpy размера (n_bands,) преобразованный в list для подачи во фронтенд 
        с выбранным спектром HSI, а также его статистические признаки и производные.
    
    Примеры
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from bio.hyper import signal
    >>> hsi = np.random.rand(100, 100, 20)  # имитирует ГСИ данные
    >>> spectre = signal(hsi, 10, 25)
    >>> plt.plot(spectre['signal'])
    >>> plt.show()
    >>> print(len(spectre['signal']))
    20  # Example output

    '''
    spectre = hsi[i, j]

    if method:
        hsi_filter = hsi_filters[method]
        spectre = hsi_filter(spectre, h)
    
    return sign_info(spectre)

def indx(hsi: np.ndarray, expr: str):
    '''
    Получение спектрального индекса над ГСИ по произвольному математическому выражению.
    Реализует парсинг математического выражения над строкой и преобразования полученной 
    математической функции в выражение над HSI каналами для синтеза нового результирующего
    канала для дальнейшего анализа.
    Функция также выполняет статистический анализ полученного канала и возвращает его
    статистические признаки и вычисляет гистограммы, возвращая результат в виде словаря.

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего гиперспектральный данные.
    expr : str
        Математической выражение в виде строки.
    
    Возвращает
    -------
    dict
        Словарь содержащий 2D массив numpy размера (n_rows, n_columns) преобразованный в list для подачи
        на фронтенд, с результирующим спектральным индексом, а также его статистические признаки и вычисленные 
        гистограммы.
    
    Примеры
    --------
    >>> import numpy as np
    >>> from bio.hyper import indx
    >>> hsi = np.random.rand(50, 50, 150)  # имитирует ГСИ данные
    >>> result = indx(hsi, '(b70 - b130) / (b70 + b130)') 
    >>> print(result['data']) # 2d array в формате питоновского списка
    '''
    channel = str2indx(hsi, expr)
    return channel


