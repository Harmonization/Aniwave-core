# Выдача данных упакованных в словари, готовых для отправки к клиенту

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import pandas as pd

from bio.hyper import rgb, signal, indx, convert_hsi, open_explorer, save_explorer
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


opened_hsi = {'name': None}
IMFs = []
cluster_info = {'centroids': [], 'medoids': []}
channel_story = {}

def apply_changes():
    '''
    Применение изменений к ГСИ
    '''

    hsi = opened_hsi['input']
    mask = opened_hsi['mask']
    
    if 'roi' in opened_hsi:
        x0, x1, y0, y1 = opened_hsi['roi']
        hsi = hsi[y0: y1, x0: x1]
        mask = mask[y0: y1, x0: x1]

    slice_lst = []
    channels_lst = opened_hsi['channels_scope']
    for scope in channels_lst:
        if len(scope) == 2:
            a, b = scope
            slice_lst.append(hsi[..., a: b])
        elif len(scope) == 1:
            slice_lst.append(hsi[..., scope])
    
    hsi = np.dstack(slice_lst)

    # if 'thr_expr' in opened_hsi and opened_hsi['thr_expr']:
    #     lower = opened_hsi['lower']
    #     upper = opened_hsi['upper']
    #     thr_expr = opened_hsi['thr_expr']
    #     channel = channel_story[thr_expr]
    #     if 'roi' in opened_hsi: channel = channel[x0: x1, y0: y1]
    #     hsi[(channel < lower) | (upper > channel)] = 0

    # Маска
    hsi[mask == 0] = 0

    opened_hsi['hsi'] = hsi
    print(opened_hsi['hsi'].shape)

def save_band(expr: str = ''):
    channel = channel_story[expr]
    # result_dct = opened_hsi['hist']

    # Заголовок спектрограммы
    header_table = True
    nm = None

    if 'raw_nm' in opened_hsi: 
        nm = opened_hsi['raw_nm']
    
    elif 'nm' in opened_hsi:
        nm = opened_hsi['nm']
        
    if not (nm is None) and channel.shape[1] == len(nm):
        header_table = nm

    path = save_explorer(expr.replace('/', ' делить на '))
    print(path)
    if path:
        with pd.ExcelWriter(''.join(path), engine='openpyxl') as writer:
            pd.DataFrame(channel).to_excel(writer, sheet_name='Индекс', header=header_table)

def get_convert():
    '''
    Функция для открытия проводника, открытия HSI и конвертации в массив np.array
    '''
    path = open_explorer()
    hsi, nm, waves, rgb_bands = convert_hsi(path)
    print(hsi.shape)
    
    name = path.split('/')[-1].split('.')[0]
    if name != opened_hsi['name']:
        opened_hsi['name'] = name
        opened_hsi['input'] = hsi
        opened_hsi['hsi'] = hsi
        opened_hsi['nm'] = nm
        opened_hsi['raw_nm'] = waves
        opened_hsi['channels_scope'] = [(0, hsi.shape[2])]
        opened_hsi['lower'] = -100000
        opened_hsi['upper'] = 100000
        opened_hsi['thr_expr'] = ''

        opened_hsi['mask'] = (np.zeros(hsi.shape[:2]) + 255).astype(np.uint8)

    # rgb_dct = get_rgb()
    return {'name': name, 'count_bands': hsi.shape[2], 'nm': nm.tolist(), 'rows': hsi.shape[0], 'cols': hsi.shape[1]} #| rgb_dct

def save_convert():
    '''
    Функция для сохранения в хранилище текущего HSI
    '''
    
    name = opened_hsi['name']
    hsi = opened_hsi['hsi']
    nm = opened_hsi['nm']
    np.save(f'Data/{name}.npy', hsi)
    np.save(f'Data/{name}_nm.npy', nm)
    return {'complete': '1'}

def download_hsi(path: str):
    '''
    Функция для загрузки ГСИ с диска в локальную сеть или на сервер.

    Параметры
    ----------
    path : str
        Путь к файлу на диске.
    '''
    
    filename = path.split('/')[-1]
    if filename != opened_hsi['name']:
        opened_hsi['input'] = download(path)

def change_channels(channels_expr: str = '0-203'):
    '''
    Изменение используемых каналов
    '''

    nm = opened_hsi['nm']
    newNm = []

    channels_lst = []
    for scope_expr in channels_expr.split(','):
        scope = list(map(int, scope_expr.split('-')))
        if len(scope) == 2:
            a, b = scope
            channels_lst.append((a, b+1))
            newNm += nm[a: b+1].tolist()
        elif len(scope) == 1:
            channels_lst.append(scope)
            newNm += [int(nm[scope])]

    opened_hsi['channels_scope'] = channels_lst

    apply_changes()
    
    print(len(newNm))
    return {'nm': newNm}

def change_roi(x0: int = 0, x1: int = 0, y0: int = 0, y1: int = 0):
    '''
    Смена координат главного ROI
    '''

    hsi = opened_hsi['input']

    if x1 == 0: x1 = hsi.shape[0]
    if y1 == 0: y1 = hsi.shape[1]

    opened_hsi['roi'] = [x0, x1, y0, y1]

    apply_changes()
    return

def change_thr(thr_expr: str, lower: float, upper: float):
    '''
    Отделение фона
    '''

    opened_hsi['lower'] = lower
    opened_hsi['upper'] = upper
    opened_hsi['thr_expr'] = thr_expr

    apply_changes()
    return

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
    hsi, nm = open(name)
    if name != opened_hsi['name']:
        opened_hsi['name'] = name
        opened_hsi['input'] = hsi
        opened_hsi['hsi'] = hsi
        opened_hsi['nm'] = nm
        opened_hsi['channels_scope'] = [(0, hsi.shape[2])]

        opened_hsi['mask'] = (np.zeros(hsi.shape[:2]) + 255).astype(np.uint8)
        
    return {'name': name, 'count_bands': hsi.shape[2], 'nm': nm.tolist(), 'rows': hsi.shape[0], 'cols': hsi.shape[1]}

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

def get_clusters(k: int, method: str):
    '''
    Обертка над кластеризацией ГСИ и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
    k : int
        Количество искомых классов.
    method : str
        Выбранный метод кластеризации в формате строки.
    
    Возвращает
    -------
    dict
        Словарь с данными результирующей сегментации (np.array) и его гистограммы.
    '''

    hsi = opened_hsi['hsi']
    segmentation = clusters(hsi, k, method)
    # hist, bins = channel_hist(segmentation, bins=np.unique(segmentation), around=False, zero_del=False)
    opened_hsi['hist'] = band_info(segmentation, zero_del=False) #{'hist': hist.tolist(), 'bins': bins.tolist()}

    cb = cv2.applyColorMap((segmentation * 50).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite('Data/segmentation.png', cb)

    # return {'segmentation': segmentation.tolist(), 'hist': hist.tolist(), 'bins': bins.tolist()}

def get_clusters_2(thr: float = .99, method: str = 'cosine', metrics: str = 'cosine'):
    '''
    Обертка над второй группой методов кластеризации ГСИ и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
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

    hsi = opened_hsi['hsi']

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

def get_cluster_corr(mode: str = 'centroids'):
    '''
    Получение матрицы корреляции и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
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

def get_reley():
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

    hsi = opened_hsi['hsi']
    spectre = rayleigh_scattering(hsi, inplace=True)
    spectre_info = [sign_info(spectre)]
    return {'reley': spectre_info}

def get_sigma(sigma: int = 2):
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

    hsi = opened_hsi['hsi']
    thresholds = np.zeros(hsi.shape[-1:], dtype=np.float32)
    changed_hsi = sigma_maximum_filter(hsi, sigma=sigma, thresholds=thresholds)
    
    spectre_info = [sign_info(thresholds)]
    return {'sigma': spectre_info}

def get_rgb_synthesize(red: int = 70, green: int = 51, blue: int = 19, red_mode: int | None = None, green_mode: int | None = None, blue_mode: int | None = None):
    '''
    Получение синтезированного RGB изображения над выбранным ГСИ и форматирование результатов 
    для передачи на сторону клиента.

    Параметры
    ----------
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
    
    hsi = opened_hsi['hsi']
    
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

def get_emd(number_of_modes: int = 8, windows_size: list[int] = [3, 3, 5]):
    '''
    Получение модального разложения над выбранным ГСИ и форматирование результатов 
    для передачи на сторону клиента.

    Параметры
    ----------
    number_of_modes : int
        Число мод необходимых к поиску.
    windows_size : list[int]
        Ширина окна модального разложения.
    
    Возвращает
    -------
    dict
        Словарь с полученным модальным разложением и его параметрами.
    '''
    
    hsi = opened_hsi['hsi']
    global IMFs
    IMFs, windows = SWEMD(hsi, number_of_modes=number_of_modes, windows_size=windows_size)
    # return IMFs
    # IMFs = emd(hsi, number_of_modes, windows_size)
    return {'emd' : {'channel': IMFs[0][..., 0].tolist(), 'number_of_modes': number_of_modes, 'windows_size': 3, 'n_band': 0, 'n_mode': 0}}

def get_emd_channel(n_mode: int = 0, n_band: int = 0):
    '''
    Получение выбранной моды и канала для найденного модального разложения и форматирование 
    результатов для передачи на сторону клиента.

    Параметры
    ----------
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

def get_spectral_classes(method: str, x: int, y: int):
    '''
    Получение каналов рассчитанных методами спектральной классификации и форматирование 
    результатов для передачи на сторону клиента.

    Параметры
    ----------
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
    
    hsi = opened_hsi['hsi']
    segmentation = spectral_classes(hsi, method, x, y)
    # hist, bins = channel_hist(segmentation, bins=20)
    return band_info(segmentation)

def get_endmembers(method: str, k: int):
    '''
    Получение результатов поиска эталонов над выбранным ГСИ и форматирование 
    результатов для передачи на сторону клиента.

    Параметры
    ----------
    method : str
        Метод построения карт.
    endmembers : list[int]
        Найденные классы, например, методом end_members.
    
    Возвращает
    -------
    dict
        Словарь с найденными чистами сигнатурами.
    '''
    
    hsi = opened_hsi['hsi']
    endmembers = end_members(hsi, method, k)
    endmembers = np.array(endmembers).ravel()
    return {'endmembers': endmembers.tolist()}

def get_amaps(method: str, endmembers: list[list[int]]):
    '''
    Получение карт изобилия над выбранным ГСИ и форматирование 
    результатов для передачи на сторону клиента.

    Параметры
    ----------
    method : str
        Метод построения карт.
    endmembers : list[int]
        Найденные классы, например, методом end_members.
    
    Возвращает
    -------
    dict
        Словарь с найденными картами изобилия.
    '''

    hsi = opened_hsi['hsi']
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

# def get_rgb(bands: tuple = (70, 51, 18)):
#     '''
#     Получение простого RGB изображения над выбранным ГСИ и форматирование 
#     результатов для передачи на сторону клиента. (Для снимков снятых со спектрографа 
#     Specim IQ).

#     Параметры
#     ----------
    
#     Возвращает
#     -------
#     dict
#         Словарь с синтезированным RGB изображением.
#     '''

#     hsi = opened_hsi['hsi']
#     return {'rgb' : rgb(hsi, bands).tolist()}

def get_signal(i: int, j: int, method: str = '', h: int = 5):
    '''
    Получение спектральной сигнатуры по указанным координатам пикселей над выбранным 
    ГСИ и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
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

    hsi = opened_hsi['input']
    return signal(hsi, i, j, method, h)

def get_indx(expr: str, t : float = 0, condition: str = ''):
    '''
    Получение спектрального индекса над ГСИ по произвольному математическому выражению над выбранным 
    ГСИ и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
    expr : str
        Математической выражение в виде строки.
    t    : float
        Нижний порог маски
    
    Возвращает
    -------
    dict
        Словарь с полученным массивом содержащим спектральный индекс.
    '''

    hsi = opened_hsi['input']
    channel = indx(hsi, expr)
    channel_story[expr] = channel
    

    cmap = cv2.COLORMAP_JET
    scale_channel = (channel * 255).astype(np.uint8)

    cb = cv2.applyColorMap(scale_channel, cmap)

    if condition: 
        scale_t = t * 255
        alpha = scale_channel.copy()

        if condition == 'lower':
            alpha[scale_channel > scale_t] = 0
            alpha[scale_channel <= scale_t] = 255
        else:
            alpha[scale_channel <= scale_t] = 0
            alpha[scale_channel > scale_t] = 255
        opened_hsi['mask'] = alpha
        apply_changes()
    
    alpha = opened_hsi['mask']

    # Вычислявем гистограмму маскированного канала
    channel_masked = channel.copy()
    channel_masked[alpha == 0] = 0
    opened_hsi['hist'] = band_info(channel_masked, zero_del=True)
    
    bgr = cv2.split(cb)
    rgba = bgr + (alpha,)
    cv2.imwrite('Data/band_0.png', cv2.merge(rgba))
    # return result_dct

def get_hist():
    return opened_hsi['hist']

def get_rgb(bands: tuple = (70, 51, 18)):
    hsi = opened_hsi['input']
    rgb = hsi[..., bands]
    scale_rgb = (rgb * 255).astype(np.uint8)
    cv2.imwrite('Data/rgb.png', scale_rgb)
    return

def get_idx_mx(name: str):
    '''
    Получение статистического спектра и его спектрограммы (в виде матрицы) над выбранным 
    ГСИ и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
    name : str
        Название статистического признака.
    
    Возвращает
    -------
    dict
        Словарь с полученным массивом содержащим результирующий спектр, матрицу и его статистику,
        включая гистограмму.
    '''

    hsi = opened_hsi['hsi']
    print(hsi.shape)
    # mask = opened_hsi['mask']
    # hsi[mask == 0] = 0
    
    res, mx = idx_mx_info(hsi, name)
    channel_story[name] = mx
    opened_hsi['hist'] = res

    # mx = res['signal']['idx_mx']['data']
    # mx = cv2.inRange(mx, -1, 1)
    mx_masked = mx.copy()
    mx_masked[mx < -1] = -1
    mx_masked[mx > 1] = 1

    cb = cv2.applyColorMap(((mx_masked - mx_masked.min())/(mx_masked.max() - mx_masked.min())*255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite('Data/mx_0.png', cb)
    # return idx_mx_info(hsi, name, startBand, endBand)

def get_regression(b1: int, b2: int):
    '''
    Получение линии регресии и графика рассяния между двумя выбранными каналами ГСИ 
    и форматирование результатов для передачи на сторону клиента.

    Параметры
    ----------
    b1 : int
        Первый канал для регрессии.
    b2 : int
        Второй канал для регрессии.
    
    Возвращает
    -------
    dict
        Словарь с полученной регрессией и точками рассеяния.
    '''

    hsi = opened_hsi['hsi']
    return regression(hsi, b1, b2)

def get_corr_mx(startBand: int = 0, endBand: int = 203):
    '''
    Получение матрицы корреляции над выбранным ГСИ и форматирование результатов 
    для передачи на сторону клиента.

    Параметры
    ----------
    startBand : int
        Начала учитываемого диапазона каналов.
    endBand : int
        Конец учитываемого диапазона каналов.
    
    Возвращает
    -------
    dict
        Словарь с полученным массивом содержащим матрицу корреляции.
    '''

    hsi = opened_hsi['hsi']
    return corr_mx(hsi[..., startBand: endBand + 1])

def get_channel_story():
    return {'save_channels': len(channel_story)}