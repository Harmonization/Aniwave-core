import numpy as np
from sklearn.cluster import KMeans, OPTICS, AffinityPropagation, Birch, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from pysptools.abundance_maps.amaps import FCLS, UCLS, NNLS
from pysptools.eea import NFINDR, PPI
from pysptools.detection.detect import ACE, CEM, GLRT, MatchedFilter, OSP

def clusters(hsi: np.ndarray, k: int, method: str):
    '''
    Реализация различных методов кластеризации применительно к изображениям ГСИ для
    получения сегментированного изображения.

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего ГСИ.
    k : int
        Количество искомых классов.
    method : str
        Выбранный метод кластеризации в формате строки.
    
    Возвращает
    -------
    np.ndarray
        Результирующее изображение сегментации разделенное на классы.
    
    Примеры
    --------
    >>> import numpy as np
    >>> from bio.segmentation import clusters
    >>> hsi = np.random.rand(100, 100, 204)  # имитирует ГСИ
    >>> result = clusters(hsi, 5, 'kmeans')
    >>> print(result.shape)
    (100, 100)  # Карта классов
    '''

    methods = {
        'kmeans': KMeans,
        'affinity_propagation': AffinityPropagation,
        'optics': OPTICS,
        'birch': Birch,
        'spectral_clustering': SpectralClustering,
        'mini_batch_kmeans': MiniBatchKMeans,
        'agglomerative_clustering': AgglomerativeClustering,
        'gaussian_mixture': GaussianMixture
    }
    cur_method = methods[method]
    if method not in ('optics', 'gaussian_mixture', 'affinity_propagation'):
        model = cur_method(n_clusters=k)
    elif method == 'optics':
        model = cur_method(min_samples=k)
    elif method == 'gaussian_mixture':
        model = cur_method(n_components=k)
    elif method == 'affinity_propagation':
        model = cur_method()

    array2d = hsi.reshape(-1, hsi.shape[2])
    segmentation = model.fit_predict(array2d).reshape(hsi.shape[0], hsi.shape[1])
    return segmentation


def SAM(hsi, U):
    '''
    Реализация метода спектральной классификации по косинусному расстоянию (Sample angle mapper method).

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего ГСИ.
    U : np.ndarray
        Набор классов для вычисления косинусного расстояния относительно их.
    
    Возвращает
    -------
    list[np.ndarray]
        Список результирующих карта расстояния от образца до остальных пикселей.
    
    Примеры
    --------
    >>> import numpy as np
    >>> from bio.segmentation import SAM
    >>> hsi = np.random.rand(100, 100, 204)  # имитирует ГСИ
    >>> U = [np.random.rand(204)]  # имитирует спектр
    >>> result = SAM(hsi, U)
    >>> print(result[0].shape)
    (100, 100)  # Карта расстояний
    '''

    rows, cols, bands = hsi.shape
    hsi_table = hsi.reshape(rows*cols,bands)
    result = []
    for u in U:
        mod_u = np.linalg.norm(u)
        mod_hsi = np.sqrt(np.sum(hsi_table**2, axis=1))
        sam = (hsi_table @ u) / (mod_hsi * mod_u)
        sam[np.isnan(sam)] = 1

        result.append(np.arccos(sam).reshape(rows, cols))
    # result.append(np.argmax(result, axis=2))
    return result

def SID(hsi, U):
    '''
    Реализация метода спектральной классификации по спектральной дивергенции 
    (Spectral information divergence method).

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего ГСИ.
    U : np.ndarray
        Набор классов для вычисления расстояния относительно их.
    
    Возвращает
    -------
    list[np.ndarray]
        Список результирующих карта расстояния от образца до остальных пикселей.
    
    Примеры
    --------
    >>> import numpy as np
    >>> from bio.segmentation import SID
    >>> hsi = np.random.rand(100, 100, 204)  # имитирует ГСИ
    >>> U = [np.random.rand(204)]  # имитирует спектр
    >>> result = SID(hsi, U)
    >>> print(result[0].shape)
    (100, 100)  # Карта расстояний
    '''

    rows, cols, bands = hsi.shape
    hsi_table = hsi.reshape(rows*cols,bands)
    result = []
    for u in U:
        mod_u = np.linalg.norm(u) # норма вектора
        mod_hsi = np.sqrt(np.sum(hsi_table**2, axis=1)) 
        norm_u = u / mod_u # нормализованный вектор
        norm_hsi = hsi_table / mod_hsi[:, np.newaxis]
        sid = np.sum(norm_hsi * np.log(norm_hsi / norm_u) + norm_u * np.log(norm_u / norm_hsi), axis=1)
        sid[np.isnan(sid)] = 0

        result.append(sid.reshape(rows, cols))
    return result

def SCA(hsi, U):
    '''
    Реализация метода спектральной классификации по корреляции 
    (Spectral correlation angle method).

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего ГСИ.
    U : np.ndarray
        Набор классов для вычисления расстояния относительно их.
    
    Возвращает
    -------
    list[np.ndarray]
        Список результирующих карта расстояния от образца до остальных пикселей.
    
    Примеры
    --------
    >>> import numpy as np
    >>> from bio.segmentation import SCA
    >>> hsi = np.random.rand(100, 100, 204)  # имитирует ГСИ
    >>> U = [np.random.rand(204)]  # имитирует спектр
    >>> result = SCA(hsi, U)
    >>> print(result[0].shape)
    (100, 100)  # Карта расстояний
    '''

    rows, cols, bands = hsi.shape
    hsi_table = hsi.reshape(rows*cols,bands)
    result = []
    for u in U:
        mean_x = np.mean(hsi_table, axis=1) # вектор n
        mean_y = np.mean(u) # скаляр
        mean_xy = np.mean(hsi_table * u, axis=1) # вектор n
        sigma_x = np.std(hsi_table, axis=1) # вектор n
        sigma_y = np.std(u) # скаляр
        sca = (mean_xy - mean_x * mean_y) / (sigma_x * sigma_y)
        sca[np.isnan(sca)] = 0

        result.append(sca.reshape(rows, cols))
    return result

def Chebychev(hsi, U):
    '''
    Реализация метода спектральной классификации по расстоянию Чебышева
    (Spectral correlation angle method).

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего ГСИ.
    U : np.ndarray
        Набор классов для вычисления расстояния относительно их.
    
    Возвращает
    -------
    list[np.ndarray]
        Список результирующих карта расстояния от образца до остальных пикселей.
    
    Примеры
    --------
    >>> import numpy as np
    >>> from bio.segmentation import Chebychev
    >>> hsi = np.random.rand(100, 100, 204)  # имитирует ГСИ
    >>> U = [np.random.rand(204)]  # имитирует спектр
    >>> result = Chebychev(hsi, U)
    >>> print(result[0].shape)
    (100, 100)  # Карта расстояний
    '''

    rows, cols, bands = hsi.shape
    hsi_table = hsi.reshape(rows*cols,bands)
    result = []
    for u in U:
        res = np.max(hsi_table - u, axis=1)
        res[np.isnan(res)] = 0

        result.append(res.reshape(rows, cols))
    return result

def Detect(hsi, U, method):
    '''
    Реализация метода спектральной классификации по расстоянию Чебышева
    (Spectral correlation angle method).

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего ГСИ.
    U : np.ndarray
        Набор классов для вычисления расстояния относительно их.
    
    Возвращает
    -------
    list[np.ndarray]
        Список результирующих карта расстояния от образца до остальных пикселей.
    
    Примеры
    --------
    >>> import numpy as np
    >>> from bio.segmentation import Chebychev
    >>> hsi = np.random.rand(100, 100, 204)  # имитирует ГСИ
    >>> U = [np.random.rand(204)]  # имитирует спектр
    >>> result = Chebychev(hsi, U)
    >>> print(result[0].shape)
    (100, 100)  # Карта расстояний
    '''

    rows, cols, bands = hsi.shape
    hsi_table = hsi.reshape(rows*cols,bands)
    result = []
    for u in U:

        res = method(hsi_table, u)
        res[np.isnan(res)] = 0

        result.append(res.reshape(rows, cols))
    return result

def spectral_classes(hsi: np.ndarray, method: str, x: int, y: int):
    '''
    Обертка над методами спектральной классификации для применения их к ГСИ 
    относительно выбранного вручную пикселя (спектральной сигнатуры) с известными 
    координатами.

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего ГСИ.
    method : str
        Метод спектральной классификации.
    x : int
        Координата ряда пикселя.
    y : int
        Координата столбца пикселя.
    
    Возвращает
    -------
    np.ndarray
        Результирующая карта расстояния от образца до остальных пикселей.
    
    Примеры
    --------
    >>> import numpy as np
    >>> from bio.segmentation import spectral_classes
    >>> hsi = np.random.rand(100, 100, 204)  # имитирует ГСИ
    >>> result = spectral_classes(hsi, 'sam', 10, 10)
    >>> print(result.shape)
    (100, 100)  # Карта расстояний
    '''

    methods_distance = {'sam': SAM, 'sid': SID, 'sca': SCA, 'chebychev': Chebychev}
    methods_detection = {'ace': ACE,'cem': CEM,'glrt': GLRT,'mf': MatchedFilter,'osp': OSP}

    if method in methods_distance:
        segmentation = methods_distance[method](hsi, [hsi[y, x]])[0]
    else:
        segmentation = Detect(hsi, [hsi[y, x]], methods_detection[method])[0]
    return segmentation

def end_members(hsi: np.ndarray, method: str, k: int) -> list[tuple[int]]:
    '''
    Реализация поиска классов на изображении HSI методами спектрального размешивания.

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего ГСИ.
    method : str
        Метод поиска.
    k : int
        Количество искомых классов.
    
    Возвращает
    -------
    np.ndarray
        Массив размера (n_classes, n_bands) найденных классов на ГСИ.
    
    Примеры
    --------
    >>> import numpy as np
    >>> from bio.segmentation import end_members
    >>> hsi = np.random.rand(100, 100, 204)  # имитирует ГСИ
    >>> result = end_members(hsi, 'ppi', 5)
    >>> print(result.shape)
    (5, 204)  # Найденные классы
    '''

    methods = {'nfindr': NFINDR, 'ppi': PPI}
    model = methods[method]()
    endmembers = model.extract(hsi, k)
    endmembers = model.get_idx()
    return endmembers

def abundance_maps(hsi: np.ndarray, method: str, endmembers: list[int]):
    '''
    Реализация методов сегментации HSI на основе карт изобилия. Для данного ГСИ и 
    известного набора классов (спектров) строится набор двумерных изображений, отображающих
    процент содержания каждого спектра в каждом пикселе.

    Параметры
    ----------
    hsi : np.ndarray
        3D массив numpy размера `(n_rows, n_columns, n_bands)` содержащего ГСИ.
    method : str
        Метод построения карт.
    endmembers : list[int]
        Найденные классы, например, методом end_members.
    
    Возвращает
    -------
    np.ndarray
        Массив размера (n_maps, n_rows, n_columns) найденных карт изобилия для ГСИ и 
        выбранных классов.
    
    Примеры
    --------
    >>> import numpy as np
    >>> from bio.segmentation import abundance_maps
    >>> hsi = np.random.rand(100, 100, 204)  # имитирует ГСИ
    >>> U = np.random.rand(5, 204)  # имитирует ГСИ
    >>> result = abundance_maps(hsi, 'fcls', U)
    >>> print(result.shape)
    (5, 100, 100)  # Найденные карты изобилия
    '''

    methods = {'fcls': FCLS, 'ucls': UCLS, 'nnls': NNLS}
    model = methods[method]
    array2d = hsi.reshape(-1, hsi.shape[2])
    members = np.array([hsi[*member] for member in endmembers])
    print(members)

    amaps = model(array2d, members)
    amaps = amaps.reshape(hsi.shape[0], hsi.shape[1], amaps.shape[1])
    return amaps