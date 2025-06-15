# Парсинг математических выражений из строки

import re

import numpy as np
from sympy import lambdify, symbols
from sympy.parsing.sympy_parser import parse_expr
from sympy.printing.latex import LatexPrinter

def str2expr(string: str) -> None:
    '''
    Преобразование математического выражения записанного в строке в функцию python,
    а также получение номеров каналов, которые были записаны особым образом в функции.

    Параметры
    ----------
    string : str
        Строка с математическим выражением, в котором участвуют каналы ГСИ.
    
    Возвращает
    -------
    list[int]
        Номера каналов участвующих в математическом выражении
    function
        Преобразованное в функцию над каналами математическое выражение
    '''
    
    # Получение функции из строки
    
    # str -> sympy-expression
    expr = parse_expr(string)

    # Поиск переменных (роль которых выполняют номера каналов, заданные в виде 'СловоЧисло', напр. 'b70' или 'канал70')
    lets = symbols(list(set(re.findall(r'[a-zA-Zа-яА-Я]+\d+', string))), integer=True, positive=True) # переменные

    # sympy-expression -> функция python
    function = lambdify(lets, expr)

    # переменные -> номера каналов
    bands: list[int] = [int(re.search(r'\d+', str(let))[0]) for let in lets]

    return bands, function

def str2nm(string: str) -> None:
    '''
    Преобразование строки для выражений использующих нанометры. Номера каналов заменяются
    нанометрами. Реализовано для анализа ГСИ полученных со спектрометра Specim IQ. 
    Трансформированная строка возвращается в формате Latex и может быть использована, 
    например, для заголовка matplotlib.

    Параметры
    ----------
    string : str
        Строка с математическим выражением, в котором участвуют каналы ГСИ.
    
    Возвращает
    -------
    string
        Выражение sympy содержащее формат latex
    '''

    # Трансформация строки: номера каналов заменяются на нанометры (таблица соответствий хранится в wave)
    
    latex_printer = LatexPrinter()

    def transform_string(string: str) -> str:
        el = string.group(0)
        return f"{re.search(r'[a-zA-Zа-яА-Я]+', el)[0]}{nm[int(re.search(r'[0-9]+', el)[0])]}"
    
    # str_band -> str_nm
    string_nm = re.sub(r'[a-zA-Zа-яА-Я]+\d+', transform_string, string)

    # str_nm -> sympy-expression
    expr_nm = parse_expr(string_nm)

    # sympy-expression -> формула latex
    latex_nm = latex_printer.doprint(expr_nm)

    return latex_nm

def expr2channel(hsi: np.ndarray[float], bands, function) -> np.ndarray[float]:
    '''
    Вычисление спектрального индекса, заданного функцией и каналами, над выбранным ГСИ.

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего гиперспектральный данные.
    bands : list[int]
        Номера каналов полученные из метода str2expr.
    function : function
        Функция полученная из метода str2expr.
    
    Возвращает
    -------
    np.ndarray
        Результирующий спектральный индекс над данными ГСИ.
    '''

    # Вычислить одноканальное изображение в соответствии с полученной функцией

    return function(*[(hsi[:, :, b] if b < 300 else hsi[..., np.where(np.array(nm) == b)[0][0]]) for b in bands])

def str2indx(hsi: np.ndarray, string: str) -> np.ndarray:
    '''
    Вычисление спектрального индекса, заданного математическим выражением, содержащим номера каналов,
    заданных особым образом.

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего гиперспектральный данные.
    string : str
        Строка с математическим выражением, в котором участвуют каналы ГСИ.
    
    Возвращает
    -------
    np.ndarray
        Результирующий спектральный индекс над данными ГСИ.
    
    Примеры
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from bio.parsing import str2indx
    >>> hsi = np.random.rand(100, 100, 20)  # имитирует ГСИ данные
    >>> result = str2indx(hsi, 'b70 + b60 / 3')
    >>> plt.imshow(result)
    >>> plt.show()
    >>> print(result.shape)
    (100, 100)  # Полученный спектральный индекс
    '''

    # Вычислить спектральный индекс определяемый выражением (string) над выбранным HSI 

    bands, function = str2expr(string)
    spectral_indx = expr2channel(hsi, bands, function)#.copy()
    # spectral_indx[np.isnan(spectral_indx) | np.isinf(spectral_indx)] = 0
    return spectral_indx

nm = [
    397,
    400,
    403,
    406,
    409,
    412,
    415,
    418,
    420,
    423,
    426,
    429,
    432,
    435,
    438,
    441,
    444,
    446,
    449,
    452,
    455,
    458,
    461,
    464,
    467,
    470,
    473,
    476,
    478,
    481,
    484,
    487,
    490,
    493,
    496,
    499,
    502,
    505,
    508,
    510,
    513,
    516,
    519,
    522,
    525,
    528,
    531,
    534,
    537,
    540,
    543,
    546,
    549,
    551,
    554,
    557,
    560,
    563,
    566,
    569,
    572,
    575,
    578,
    581,
    584,
    587,
    590,
    593,
    596,
    599,
    602,
    605,
    607,
    610,
    613,
    616,
    619,
    622,
    625,
    628,
    631,
    634,
    637,
    640,
    643,
    646,
    649,
    652,
    655,
    658,
    661,
    664,
    667,
    670,
    673,
    676,
    679,
    682,
    685,
    688,
    691,
    694,
    697,
    700,
    703,
    706,
    709,
    712,
    715,
    718,
    721,
    724,
    727,
    730,
    733,
    736,
    739,
    742,
    745,
    748,
    751,
    754,
    757,
    760,
    763,
    766,
    769,
    772,
    775,
    778,
    781,
    784,
    787,
    790,
    793,
    796,
    799,
    802,
    805,
    808,
    811,
    814,
    817,
    820,
    823,
    826,
    829,
    832,
    835,
    838,
    841,
    844,
    847,
    850,
    853,
    856,
    859,
    862,
    866,
    869,
    872,
    875,
    878,
    881,
    884,
    887,
    890,
    893,
    896,
    899,
    902,
    905,
    908,
    911,
    914,
    917,
    920,
    924,
    927,
    930,
    933,
    936,
    939,
    942,
    945,
    948,
    951,
    954,
    957,
    960,
    963,
    967,
    970,
    973,
    976,
    979,
    982,
    985,
    988,
    991,
    994,
    997,
    1000,
    1004
]