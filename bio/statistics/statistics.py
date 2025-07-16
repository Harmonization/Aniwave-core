import numpy as np
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression

# features = ['max', 'min', 'mean', 'std', 'scope', 'iqr', 'entropy', 'q1', 'median', 'q3']

def channel_hist(channel: np.ndarray, bins: int = 20, around: bool = True, zero_del: bool = True):
    '''
    Вычисление гистограммы для канала или спектрального индекса ГСИ с учетом его маски.

    Параметры
    ----------
    channel : np.ndarray
        2D массив numpy размера `(n_rows, n_columns)` содержащего одноканальное изображение (индекс).
    bins : int
        Количество бинов гистограммы.
    around : bool
        Нужно ли округлять значения гистограммы для более понятного отображения со стороны клиента.
    zero_del : bool
        Нужно ли учитывать нулевые значения (нужно для учета маски изображения).
    
    Возвращает
    -------
    np.ndarray
        Массив значений гистограммы
    np.ndarray
        Массив бинов гистограммы
    '''

    # Гистограмма
    array1d = np.copy(channel[channel != 0] if zero_del else channel.ravel())
    q5, q95 = np.percentile(array1d, (5, 95))
    array1d[array1d < q5] = 0
    array1d[array1d > q95] = 0
    
    hist, bins = np.histogram(array1d, bins=bins)
    if around: bins = np.around(bins.astype(float), 2)
    return hist, bins

def sign_stat(sign: np.ndarray) -> dict[str, float | np.ndarray]:
    '''
    Вычисление статистических показателей выбранной спектральной сигнатуры.

    Параметры
    ----------
    sign : np.ndarray
        1D массив numpy размера `(n_bands)` содержащего спектральную сигнатуру.
    
    Возвращает
    -------
    dict[str, float | np.ndarray]
        Словарь, содержащий данные спектральной сигнатуры и ее статистические значения.
    '''

    # Вычисление статистики спектра
    data_dict = {
        'max': sign.max(),
        'min': sign.min(),
        'mean': sign.mean(),
        'std': sign.std(),
        'scope': sign.max() - sign.min(),
        'iqr': np.subtract(*np.percentile(sign, [75, 25])),
        'q1': np.percentile(sign, 25),
        'median': np.median(sign),
        'q3': np.percentile(sign, 75),
        'entropy': entropy(np.unique(sign, return_counts=True)[1])
    }
    data_dict = {k: round(float(v), 3) for k, v in data_dict.items()}
    data_dict['sign'] = np.around(sign, 3).tolist()
    return data_dict

def sign_diff(sign: np.ndarray) -> np.ndarray:
    '''
    Вычисление производной выбранной спектральной сигнатуры.

    Параметры
    ----------
    sign : np.ndarray
        1D массив numpy размера `(n_bands)` содержащий спектральную сигнатуру.
    
    Возвращает
    -------
    np.ndarray
        1D массив numpy размера `(n_bands)` содержащий производную исходного спектра.
    '''

    # Вычисление производной спектра
    dx = 1 / sign.shape[0]
    return np.diff(sign) / dx

def sign_info(sign: np.ndarray):
    '''
    Вычисление производной и статистических показателей выбранной спектральной сигнатуры.

    Параметры
    ----------
    sign : np.ndarray
        1D массив numpy размера `(n_bands)` содержащий спектральную сигнатуру.
    
    Возвращает
    -------
    dict
        Словарь, содержащий данные спектральной сигнатуры, ее статистические значения,
        а также производную и ее статистические значения.
    '''

    # Вычисление стат величин для спектра и его производной
    diff = sign_diff(sign)
    return {
        'signal': sign_stat(sign),
        'diff': sign_stat(diff)
    }

def calc_entropy(hsi: np.ndarray, axis=None):
    '''
    Вычисление энтропии либо над целым ГСИ, или над каждым каналом в отдельности.

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего гиперспектральный данные.
    axis : tuple | None
        Оси, по которым вычисляется энтропия. Если axis=None, то вычисляется над целым ГСИ.
    
    Возвращает
    -------
    float | np.ndarray
        Рассчитанный показатель энтропии, или numpy массив показателей размера (n_bands,)
    '''

    # Вычисление энтропии над hsi или над каждым каналом в отдельности (если axis не None)
    if axis is None:
        mask = ~np.isnan(hsi) & (hsi != 0)
        return entropy(np.unique(hsi[mask], return_counts=True)[1])
    else:
        res = []
        for band in range(hsi.shape[-1]):
            channel = hsi[..., band]
            mask = ~np.isnan(channel) & (channel != 0)
            uniq = np.unique(channel[mask], return_counts=True)[1]
            res.append(entropy(uniq))
        return res
    
def calc_max(data: np.ndarray, axis: tuple | None = None):
    '''
    Быстрое вычисления максимума с учетом маски по выбранным осям (т.е для данных полностью или по отдельным осям).

    Параметры
    ----------
    hsi : np.ndarray
        Массив numpy любого размера, содержащего гиперспектральные данные.
    axis : tuple | None
        Оси, по которым вычисляется статистический признак. 
    
    Возвращает
    -------
    float | np.ndarray
        Рассчитанный показатель, или numpy массив показателей.
    '''

    mask = ~np.isnan(data) & (data != 0)
    return data.max(axis=axis, initial=-10, where=mask)

def calc_min(data: np.ndarray, axis: tuple | None = None):
    '''
    Быстрое вычисления минимума с учетом маски по выбранным осям (т.е для данных полностью или по отдельным осям).

    Параметры
    ----------
    hsi : np.ndarray
        Массив numpy любого размера, содержащего гиперспектральные данные.
    axis : tuple | None
        Оси, по которым вычисляется статистический признак. 
    
    Возвращает
    -------
    float | np.ndarray
        Рассчитанный показатель, или numpy массив показателей.
    '''
    mask = ~np.isnan(data) & (data != 0)
    return data.min(axis=axis, initial=10, where=mask)

def calc_mean(data: np.ndarray, axis: tuple | None = None):
    '''
    Быстрое вычисления среднего значения с учетом маски по выбранным осям (т.е для данных полностью или по отдельным осям).

    Параметры
    ----------
    hsi : np.ndarray
        Массив numpy любого размера, содержащего гиперспектральные данные.
    axis : tuple | None
        Оси, по которым вычисляется статистический признак. 
    
    Возвращает
    -------
    float | np.ndarray
        Рассчитанный показатель, или numpy массив показателей.
    '''
    mask = ~np.isnan(data) & (data != 0)
    return data.mean(axis=axis, where=mask)

def calc_std(data: np.ndarray, axis: tuple | None = None):
    '''
    Быстрое вычисления среднеквадратического отклонения с учетом маски по выбранным осям 
    (т.е для данных полностью или по отдельным осям).

    Параметры
    ----------
    hsi : np.ndarray
        Массив numpy любого размера, содержащего гиперспектральные данные.
    axis : tuple | None
        Оси, по которым вычисляется статистический признак. 
    
    Возвращает
    -------
    float | np.ndarray
        Рассчитанный показатель, или numpy массив показателей.
    '''
    mask = ~np.isnan(data) & (data != 0)
    return data.std(axis=axis, where=mask)

def calc_scope(data: np.ndarray, axis: tuple | None = None):
    '''
    Быстрое вычисления размаха с учетом маски по выбранным осям 
    (т.е для данных полностью или по отдельным осям).

    Параметры
    ----------
    hsi : np.ndarray
        Массив numpy любого размера, содержащего гиперспектральные данные.
    axis : tuple | None
        Оси, по которым вычисляется статистический признак. 
    
    Возвращает
    -------
    float | np.ndarray
        Рассчитанный показатель, или numpy массив показателей.
    '''
    return calc_max(data, axis) - calc_min(data, axis)

def calc_median(data: np.ndarray, axis: tuple | None = None):
    '''
    Быстрое вычисления медианы с учетом маски по выбранным осям 
    (т.е для данных полностью или по отдельным осям).

    Параметры
    ----------
    hsi : np.ndarray
        Массив numpy любого размера, содержащего гиперспектральные данные.
    axis : tuple | None
        Оси, по которым вычисляется статистический признак. 
    
    Возвращает
    -------
    float | np.ndarray
        Рассчитанный показатель, или numpy массив показателей.
    '''
    mask = ~np.isnan(data) & (data != 0)
    data_without_zero = np.ma.array(data, mask=~mask)
    return np.ma.median(data_without_zero, axis=axis)

def calc_q1(data: np.ndarray, axis: tuple | None = None):
    '''
    Быстрое вычисления первого квартиля с учетом маски по выбранным осям 
    (т.е для данных полностью или по отдельным осям).

    Параметры
    ----------
    hsi : np.ndarray
        Массив numpy любого размера, содержащего гиперспектральные данные.
    axis : tuple | None
        Оси, по которым вычисляется статистический признак. 
    
    Возвращает
    -------
    float | np.ndarray
        Рассчитанный показатель, или numpy массив показателей.
    '''
    mask = ~np.isnan(data) & (data != 0)
    data_without_zero = data.copy()
    data_without_zero[~mask] = 0 # np.nan
    return np.nanpercentile(data_without_zero, q=25, axis=axis)

def calc_q3(data: np.ndarray, axis: tuple | None = None):
    '''
    Быстрое вычисления третьего квартиля с учетом маски по выбранным осям 
    (т.е для данных полностью или по отдельным осям).

    Параметры
    ----------
    hsi : np.ndarray
        Массив numpy любого размера, содержащего гиперспектральные данные.
    axis : tuple | None
        Оси, по которым вычисляется статистический признак. 
    
    Возвращает
    -------
    float | np.ndarray
        Рассчитанный показатель, или numpy массив показателей.
    '''
    mask = ~np.isnan(data) & (data != 0)
    data_without_zero = data.copy()
    data_without_zero[~mask] = 0 # np.nan 
    return np.nanpercentile(data_without_zero, q=75, axis=axis)

def calc_iqr(data: np.ndarray, axis: tuple | None = None):
    '''
    Быстрое вычисления межквартильного размаха с учетом маски по выбранным осям 
    (т.е для данных полностью или по отдельным осям).

    Параметры
    ----------
    hsi : np.ndarray
        Массив numpy любого размера, содержащего гиперспектральные данные.
    axis : tuple | None
        Оси, по которым вычисляется статистический признак. 
    
    Возвращает
    -------
    float | np.ndarray
        Рассчитанный показатель, или numpy массив показателей.
    '''
    return calc_q3(data, axis) - calc_q1(data, axis)


func_dict = {
    'max': calc_max, 
    'min': calc_min,
    'mean': calc_mean, 
    'std': calc_std, 
    'scope': calc_scope,
    'iqr': calc_iqr,
    'entropy': calc_entropy,
    'q1': calc_q1,
    'median': calc_median,
    'q3': calc_q3
}

def band_stat(hsi, name: str, startBand: int = 0, endBand: int = 203):
    '''
    Быстрое вычисление выбранного статистического признака с учетом маски по всему ГСИ 
    либо каналу, для выбранного диапазона каналов.

    Параметры
    ----------
    hsi : np.ndarray
        Массив numpy любого размера, содержащего гиперспектральные данные.
    name : str
        Имя статистического признака. 
    startBand : int
        Начало выбранного диапазона каналов для расчета. 
    endBand : int
        Конец выбранного диапазона каналов для расчета. 
    
    Возвращает
    -------
    float
        Рассчитанный статистический показатель.
    '''

    feature = func_dict[name]
    data = hsi[..., startBand: endBand + 1]

    res = feature(data)
    res = round(float(res), 3)
    return res

def band_info(band: np.ndarray, zero_del: bool = True):
    '''
    Быстрое вычисление всех статистических признаков с учетом маски по выбранному 
    каналу ГСИ, а также вычисление его гистограммы.

    Параметры
    ----------
    hsi : np.ndarray
        Массив numpy любого размера, содержащего гиперспектральные данные.
    zero_del : bool
        Нужно ли учитывать нулевые значения (нужно для учета маски изображения).
    
    Возвращает
    -------
    dict
        Словарь содержащий исходные данные, гистограмму, ее бины, а также статистические показатели.
    '''

    # 
    feature_dict = {feature: band_stat(band, feature) for feature in func_dict}
    feature_dict = {feature: float(np.around(value, 3).astype(float)) for feature, value in feature_dict.items()}
    
    hist, bins = channel_hist(band, zero_del=zero_del)
    feature_dict['hist'] = hist.tolist()
    feature_dict['bins'] = bins.tolist()
    # feature_dict['data'] = np.around(band, 3).tolist()
    
    return feature_dict

def idx_matrix(spectre: np.ndarray[float]) -> np.ndarray[float]:
    '''
    Быстрое вычисление спектрограммы для данного спектра, содержащую 
    нормализованную разницу его компонент.

    Параметры
    ----------
    spectre : np.ndarray
        Массив numpy размера (n_band,), содержащий спектральную сигнатуру.
    
    Возвращает
    -------
    np.ndarray
        Массив numpy размера (n_band, n_band), содержащий рассчитанную спектрограмму.
    '''

    # Вычисление индексной матрицы
    x, y = np.meshgrid(spectre, spectre)
    mx = np.fromfunction(lambda i, j: (x-y) / (x+y), (204, 204))
    mx[np.isnan(mx) | np.isinf(mx)] = 0
    return mx

def idx_mx_info(hsi, name: str):
    '''
    Быстрое вычисление статистического спектра (содержащего статистику каждого
    канала в отдельности) и его спектрограммы, а также вычисляет их статистические
    показатели и гистограммы.

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего гиперспектральный данные.
    name : str
        Название вычисляемого статистического признака.
    
    Возвращает
    -------
    dict
        Словарь, содержащий статистический спектр, спектрограмму, и их статистически показатели 
        и гистограммы.
    '''

    feature = func_dict[name]
    # data = hsi[..., startBand: endBand + 1]

    sign = np.array(feature(hsi, axis=(0, 1))).astype(float)
    data_dict = sign_info(sign)

    signal = data_dict['signal']['sign']
    # diff = data_dict['diff']['sign']

    mx = idx_matrix(signal)
    # hist, bins = channel_hist(mx, zero_del=False)
    # data_dict['signal']['idx_mx'] = {'hist': np.around(hist, 3).tolist(), 'bins': bins.tolist()}
    # data_dict['hist'] = np.around(hist, 3).tolist()
    # data_dict['bins'] = bins.tolist()

    # mx = idx_matrix(diff)
    # hist, bins = channel_hist(mx, zero_del=False)
    # data_dict['diff']['idx_mx'] = {'data': np.around(mx, 3).astype(float).tolist(), 'hist': np.around(hist, 3).tolist(), 'bins': bins.tolist()}
    
    return data_dict | band_info(mx, zero_del=False), np.around(mx, 3).astype(float)

def regression(hsi: np.ndarray, b1: int, b2: int):
    '''
    Быстрое вычисление линейной регрессии между двумя выбранными каналами ГСИ, а также
    вычисление их парных статистических показателей.

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего гиперспектральный данные.
    b1 : int
        Номер первого канала ГСИ для вычисления регрессии.
    b2 : int
        Номер второго канала ГСИ для вычисления регрессии.
    
    Возвращает
    -------
    dict
        Словарь, содержащий коеффициенты линейной регрессии, координаты точек каналов по осям для создания
        диаграммы рассеяния, а также их статистические показатели регрессии.
    '''
    
    model = LinearRegression()
    band_1, band_2 = hsi[..., b1], hsi[..., b2]
    model.fit(band_1.reshape(-1, 1), band_2.reshape(-1, 1))

    arr_1, arr_2 = band_1.ravel(), band_2.ravel()
    a, b = model.coef_[0], model.intercept_
    x = np.linspace(0, band_1.max(), 10)
    y = a * x + b
    r = np.corrcoef(arr_1, arr_2)[0, 1]
    R = r**2
    mean_elastic = a * band_1.mean() / band_2.mean()
    beta = a * band_1.std() / band_2.std()

    indx = np.random.choice(len(arr_1), len(arr_1)//10)
    points_1 = np.around(arr_1[indx], 3).tolist()
    points_2 = np.around(arr_2[indx], 3).tolist()
    return {'b1': b1, 
            'b2': b2, 
            'line_1': x.tolist(), 
            'line_2': y.tolist(), 
            'a': round(float(a), 3), 
            'b': round(float(b), 3),
            'points_1': points_1,
            'points_2': points_2,
            'correlation': round(float(r), 3), 
            'determination': round(float(R), 3), 
            'elastic': round(float(mean_elastic), 3), 
            'beta': round(float(beta), 3)}

def corr_mx(hsi: np.ndarray):
    '''
    Быстрое вычисление матрицы корреляции между каналами по выбранному ГСИ, а также
    расчет ее статистических показателей.

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего гиперспектральный данные.
    
    Возвращает
    -------
    dict
        Словарь, содержащий матрицу коореляции каналов ГСИ, а также рассчитанные статистические показатели.
    '''

    array2d = hsi.reshape(-1, hsi.shape[2]).T # shape = (204, N)
    mx = np.corrcoef(array2d)
    # hist, bins = channel_hist(mx, zero_del=False)
    return band_info(mx, zero_del=False)
    
    # return {'data': np.around(mx, 3).tolist(), 'hist': hist.tolist(), 'bins': bins.tolist(), 'info': mx_info}