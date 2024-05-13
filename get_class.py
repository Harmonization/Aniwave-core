import numpy as np
from sklearn.cluster import KMeans, OPTICS, AffinityPropagation, Birch, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from pysptools.abundance_maps.amaps import FCLS, UCLS, NNLS
from pysptools.eea import NFINDR, PPI

from algorithms.classification.sam import SAM
from algorithms.classification.sid import SID
from algorithms.classification.sca import SCA

def get_classification(hsi: np.ndarray, algorithm: str, x: int, y: int):
    algorithms = {'sam': SAM, 'sid': SID, 'sca': SCA}
    angle_maps = algorithms[algorithm](hsi, [hsi[y, x]])
    return {'x': angle_maps[0].tolist()}

def get_clusterization(hsi, k: int, method: str):
    cluster_methods = {
        'kmeans': KMeans,
        'affinity_propagation': AffinityPropagation,
        'optics': OPTICS,
        'birch': Birch,
        'spectral_clustering': SpectralClustering,
        'mini_batch_kmeans': MiniBatchKMeans,
        'agglomerative_clustering': AgglomerativeClustering,
        'gaussian_mixture': GaussianMixture
    }
    
    cur_method = cluster_methods[method]
    if method not in ('optics', 'gaussian_mixture', 'affinity_propagation'):
        model = cur_method(n_clusters=k)
    elif method == 'optics':
        model = cur_method(min_samples=k)
    elif method == 'gaussian_mixture':
        model = cur_method(n_components=k)
    elif method == 'affinity_propagation':
        model = cur_method()

    hsi_table = hsi.reshape(-1, 204)
    results = model.fit_predict(hsi_table).reshape(hsi.shape[0], hsi.shape[1])
    
    return {'classes': results.tolist()}

def get_extraction(hsi, method: str, amap: str, k: int):
    methods = {
        'nfindr': NFINDR,
        'ppi': PPI
    }

    amaps = {
        'fcls': FCLS,
        'ucls': UCLS,
        'nnls': NNLS
    }
    hsi_table = hsi.reshape(hsi.shape[0] * hsi.shape[1], hsi.shape[2])

    extract_model = methods[method]()
    endmembers = extract_model.extract(hsi, k)

    amap_model = amaps[amap]
    result = amap_model(hsi_table, endmembers)
    maps = np.around(result.reshape(hsi.shape[0], hsi.shape[1], result.shape[1]), 3)
    return {'endmembers': np.around(endmembers, 3).tolist(), 'amaps': [maps[:, :, i].tolist() for i in range(maps.shape[2])]}