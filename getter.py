# Выдача данных упакованных в словари, готовых для отправки к клиенту

import numpy as np

from bio.hyper import rgb, signal, indx
from bio.thermal import open_tir
from bio.statistics import channel_hist, idx_mx_info, regression, corr_mx, band_info
from bio.segmentation import clusters, spectral_classes, end_members, abundance_maps
from loader import download, open
from bio.statistics import sign_info

from hsip.rgb.rgb import hsi_synthesize_rgb, simple_synthesize_rgb
from hsip.analysis.analysis import get_cross_correlation_matrix
from hsip.processing.processing import rayleigh_scattering, sigma_maximum_filter
from hsip.clustering.clustering import CosClust, HDBSCAN, SCH
from hsip.swemd.swemd import SWEMD

opened_hsi = {}
IMFs = []
cluster_info = {'centroids': [], 'medoids': []}

def download_hsi(path: str):
    '''
    Функция для загрузки ГСИ с диска в локальную сеть или на сервер.

    Параметры
    ----------
    path : str
        Путь к файлу на диске.
    '''
    
    filename = path.split('/')[-1]
    if filename not in opened_hsi:
        opened_hsi[filename] = download(path)

def open_hsi(name: str):
    '''
    Открытие ГСИ на стороне сервера.

    Параметры
    ----------
    path : str
        Путь к файлу на сервере.
    
    Возвращает
    -------
    dict
        Словарь с именем открытого ГСИ и кол-во каналов.
    '''

    if name not in opened_hsi:
        opened_hsi[name] = open(name)
        
    return {'name': name, 'count_bands': opened_hsi[name].shape[2]}

def get_tir(name):
    '''
    Открытие температурного изображения на стороне сервера.

    Параметры
    ----------
    name : str
        Имя файла на сервере.
    
    Возвращает
    -------
    dict
        Словарь с данными термального изображения (np.array) и его гистограммой.
    '''

    # Словарь данных TIR
    path = f'Data/{name}'
    tir = open_tir(path)
    hist, bins = channel_hist(tir)
    return {'tir': {'data': tir.tolist(), 'hist': hist.tolist(), 'bins': bins.tolist()}}

def get_clusters(name_hsi: str, k: int, method: str):
    '''
    Обертка над кластеризацией ГСИ и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    k : int
        Количество искомых классов.
    method : str
        Выбранный метод кластеризации в формате строки.
    
    Возвращает
    -------
    dict
        Словарь с данными результирующей сегментации (np.array) и его гистограммы.
    '''

    hsi = opened_hsi[name_hsi]
    segmentation = clusters(hsi, k, method)
    hist, bins = channel_hist(segmentation, bins=np.unique(segmentation), around=False, zero_del=False)
    return {'segmentation': segmentation.tolist(), 'hist': hist.tolist(), 'bins': bins.tolist()}

def get_clusters_2(name_hsi: str, thr: float = .99, method: str = 'cosine', metrics: str = 'cosine'):
    '''
    Обертка над второй группой методов кластеризации ГСИ и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    thr : float | int
        Порог или количество искомых классов.
    method : str
        Выбранный метод кластеризации в формате строки.
    metrics : str
        Выбранная метрика кластеризации.
    
    Возвращает
    -------
    dict
        Словарь с данными результирующей сегментации (np.array) и его гистограммы, центроидами и медоидами.
    '''

    hsi = opened_hsi[name_hsi]

    r, c, b = hsi.shape
    hsi_table = hsi.reshape((r*c, b))

    if method == 'cosine':
        model = CosClust(threshold=thr, verbose=True)
    elif method == 'hdbscan':
        model = HDBSCAN(min_cluster_size=thr)
    else:
        model = SCH()

    labels = model.fit(hsi_table)
    segmentation = labels.reshape((r, c))
    centroids, medoids = model.centroids, model.medoids

    # segmentation, centroids, medoids = clusters_2(hsi, thr, method, metrics)
    hist, bins = channel_hist(segmentation, bins=np.unique(segmentation), around=False, zero_del=False)
    
    cluster_info['centroids'] = centroids
    cluster_info['medoids'] = medoids

    centroids = [sign_info(centroid) for centroid in centroids]
    medoids = [sign_info(medoid) for medoid in medoids]
    return {'segmentation': segmentation.tolist(), 'hist': hist.tolist(), 'bins': bins.tolist(), 'centroids': centroids, 'medoids': medoids}

def get_cluster_corr(name_hsi: str, mode: str = 'centroids'):
    '''
    Получение матрицы корреляции и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    mode : str
        Режим построение матрицы: для центроидов (centroids) или медоидов (medoids).
    
    Возвращает
    -------
    dict
        Словарь с данными результирующей корреляции (np.array) и его гистограммы.
    '''

    # Матрица корреляции
    spectres = cluster_info['centroids'] if mode == 'centroids' else cluster_info['medoids'][:16]
    cross_corr_mat = get_cross_correlation_matrix(spectres)
    hist, bins = channel_hist(cross_corr_mat, bins=len(spectres)-1)
    return {'segmentation': cross_corr_mat.tolist(), 'mode': mode, 'hist': hist.tolist(), 'bins': bins.tolist()}

def get_reley(name_hsi: str):
    '''
    Получение релеевского рассеяния над выбранным ГСИ и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    
    Возвращает
    -------
    dict
        Словарь с данными результирующим рассеянием (np.array), его статистических данных и его гистограммы.
    '''

    hsi = opened_hsi[name_hsi]
    spectre = rayleigh_scattering(hsi, inplace=True)
    spectre_info = [sign_info(spectre)]
    return {'reley': spectre_info}

def get_sigma(name_hsi: str, sigma: int = 2):
    '''
    Получение сигма-отклонения над выбранным ГСИ и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    sigma : int
        Число параметра сигма.
    
    Возвращает
    -------
    dict
        Словарь с данными результирующим сигма-отклонением (np.array), его статистических данных.
    '''

    hsi = opened_hsi[name_hsi]
    thresholds = np.zeros(hsi.shape[-1:], dtype=np.float32)
    changed_hsi = sigma_maximum_filter(hsi, sigma=sigma, thresholds=thresholds)
    
    spectre_info = [sign_info(thresholds)]
    return {'sigma': spectre_info}

def get_rgb_synthesize(name_hsi: str, red: int = 70, green: int = 51, blue: int = 19, red_mode: int | None = None, green_mode: int | None = None, blue_mode: int | None = None):
    '''
    Получение синтезированного RGB изображения над выбранным ГСИ и форматирование результатов 
    для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    red : int
        Номер красного канала.
    green : int
        Номер зеленого канала.
    blue : int
        Номер синего канала.
    red_mode : int
        Номер красной моды.
    green_mode : int
        Номер зеленой моды.
    blue_mode : int
        Номер синей моды.
    
    Возвращает
    -------
    dict
        Словарь с синтезированным RGB-изображением.
    '''
    
    hsi = opened_hsi[name_hsi]
    
    r = red if (red_mode is None) else IMFs[red_mode][..., red]
    g = green if (green_mode is None) else IMFs[green_mode][..., green]
    b = blue if (blue_mode is None) else IMFs[blue_mode][..., blue]

    if type(r) == int and type(g) == int and type(b) == int:
        # rgb = synthesize_rgb(hsi, r, g, b)
        rgb_bands = [r, g, b]  # Example red, green, blue bands
        rgb = hsi_synthesize_rgb(hsi, rgb_bands=rgb_bands)
    else:
        if type(r) == int:
            r = hsi[..., r]
        if type(g) == int:
            g = hsi[..., g]
        if type(b) == int:
            b = hsi[..., b]
        rgb = simple_synthesize_rgb([r, g, b])
    return {'rgb' : rgb.tolist()}

def get_emd(name_hsi: str, number_of_modes: int = 8, windows_size: list[int] = [3, 3, 5]):
    '''
    Получение модального разложения над выбранным ГСИ и форматирование результатов 
    для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    number_of_modes : int
        Число мод необходимых к поиску.
    windows_size : list[int]
        Ширина окна модального разложения.
    
    Возвращает
    -------
    dict
        Словарь с полученным модальным разложением и его параметрами.
    '''
    
    hsi = opened_hsi[name_hsi]
    global IMFs
    IMFs, windows = SWEMD(hsi, number_of_modes=number_of_modes, windows_size=windows_size)
    # return IMFs
    # IMFs = emd(hsi, number_of_modes, windows_size)
    return {'emd' : {'channel': IMFs[0][..., 0].tolist(), 'number_of_modes': number_of_modes, 'windows_size': 3, 'n_band': 0, 'n_mode': 0}}

def get_emd_channel(name_hsi: str, n_mode: int = 0, n_band: int = 0):
    '''
    Получение выбранной моды и канала для найденного модального разложения и форматирование 
    результатов для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    n_mode : int
        Номер моды.
    n_band : int
        Номер канала.
    
    Возвращает
    -------
    dict
        Словарь с выбранным модальным каналом и его гистограммой.
    '''
    
    segmentation = IMFs[n_mode][..., n_band]
    hist, bins = channel_hist(segmentation, bins=20)
    return {'emd' : {'channel': segmentation.tolist(), 'n_band': n_band, 'n_mode': n_mode, 'hist': hist.tolist(), 'bins': bins.tolist()}}

def get_spectral_classes(name_hsi: str, method: str, x: int, y: int):
    '''
    Получение каналов рассчитанных методами спектральной классификации и форматирование 
    результатов для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    method : str
        Метод спектральной классификации.
    x : int
        Координата ряда пикселя.
    y : int
        Координата столбца пикселя.
    
    Возвращает
    -------
    dict
        Словарь с результирующим каналом.
    '''
    
    hsi = opened_hsi[name_hsi]
    segmentation = spectral_classes(hsi, method, x, y)
    # hist, bins = channel_hist(segmentation, bins=20)
    return band_info(segmentation)
    # return {'segmentation': segmentation.tolist(), 'hist': hist.tolist(), 'bins': bins.tolist()}

def get_endmembers(name_hsi: str, method: str, k: int):
    '''
    Получение результатов поиска эталонов над выбранным ГСИ и форматирование 
    результатов для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    method : str
        Метод построения карт.
    endmembers : list[int]
        Найденные классы, например, методом end_members.
    
    Возвращает
    -------
    dict
        Словарь с найденными чистами сигнатурами.
    '''
    
    hsi = opened_hsi[name_hsi]
    endmembers = end_members(hsi, method, k)
    endmembers = np.array(endmembers).ravel()
    return {'endmembers': endmembers.tolist()}

def get_amaps(name_hsi: str, method: str, endmembers: list[list[int]]):
    '''
    Получение карт изобилия над выбранным ГСИ и форматирование 
    результатов для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    method : str
        Метод построения карт.
    endmembers : list[int]
        Найденные классы, например, методом end_members.
    
    Возвращает
    -------
    dict
        Словарь с найденными картами изобилия.
    '''

    hsi = opened_hsi[name_hsi]
    members = list(map(int, endmembers[0].split(',')))
    members = np.array(members).reshape(-1, 2)[:, ::-1]
    print(members)
    amaps = abundance_maps(hsi, method, members)
    amaps = np.around(amaps, 3)

    data_maps = []
    for i in range(amaps.shape[2]):
        segmentation = amaps[:, :, i]
        hist, bins = channel_hist(segmentation, bins=20)
        data_maps.append({'data': segmentation.tolist(), 'hist': hist.tolist(), 'bins': bins.tolist()})
        
    return {'amaps': data_maps}

def get_rgb(name_hsi: str):
    '''
    Получение простого RGB изображения над выбранным ГСИ и форматирование 
    результатов для передачи на сторону клиента. (Для снимков снятых со спектрографа 
    Specim IQ).

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    
    Возвращает
    -------
    dict
        Словарь с синтезированным RGB изображением.
    '''

    hsi = opened_hsi[name_hsi]
    return {'rgb' : rgb(hsi).tolist()}

def get_signal(name_hsi: str, i: int, j: int, method: str = '', h: int = 5):
    '''
    Получение спектральной сигнатуры по указанным координатам пикселей над выбранным 
    ГСИ и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
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
        Словарь с полученным массивом содержащим спектральную сигнатуру.
    '''

    hsi = opened_hsi[name_hsi]
    return signal(hsi, i, j, method, h)

def get_indx(name_hsi: str, expr: str):
    '''
    Получение спектрального индекса над ГСИ по произвольному математическому выражению над выбранным 
    ГСИ и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    expr : str
        Математической выражение в виде строки.
    
    Возвращает
    -------
    dict
        Словарь с полученным массивом содержащим спектральный индекс.
    '''

    hsi = opened_hsi[name_hsi]
    return indx(hsi, expr)

def get_idx_mx(name_hsi: str, name: str, startBand: int = 0, endBand: int = 203):
    '''
    Получение статистического спектра и его спектрограммы (в виде матрицы) над выбранным 
    ГСИ и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    name : str
        Название статистического признака.
    startBand : int
        Начала учитываемого диапазона каналов.
    endBand : int
        Конец учитываемого диапазона каналов.
    
    Возвращает
    -------
    dict
        Словарь с полученным массивом содержащим результирующий спектр, матрицу и его статистику,
        включая гистограмму.
    '''

    hsi = opened_hsi[name_hsi]
    return idx_mx_info(hsi, name, startBand, endBand)

def get_regression(name_hsi: str, b1: int, b2: int):
    '''
    Получение линии регресии и графика рассяния между двумя выбранными каналами ГСИ 
    и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    b1 : int
        Первый канал для регрессии.
    b2 : int
        Второй канал для регрессии.
    
    Возвращает
    -------
    dict
        Словарь с полученной регрессией и точками рассеяния.
    '''

    hsi = opened_hsi[name_hsi]
    return regression(hsi, b1, b2)

def get_corr_mx(name_hsi: str, startBand: int = 0, endBand: int = 203):
    '''
    Получение матрицы корреляции над выбранным ГСИ и форматирование результатов 
    для передачи на сторону клиента.

    Параметры
    ----------
    name_hsi : str
        Имя открытого ГСИ.
    startBand : int
        Начала учитываемого диапазона каналов.
    endBand : int
        Конец учитываемого диапазона каналов.
    
    Возвращает
    -------
    dict
        Словарь с полученным массивом содержащим матрицу корреляции.
    '''

    hsi = opened_hsi[name_hsi]
    return corr_mx(hsi[..., startBand: endBand + 1])