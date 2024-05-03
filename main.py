from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import yadisk

from loader import loader, download_from_path
from parse import Parser
from algorithms.statistics.stat import Statistics

TOKEN = 'y0_AgAAAAAsGvHOAApsKgAAAADrhnQx1pqpt1zjQPCSV28z1eLp1OYhH6g'
# main_root = 'Datasets and Program/HSI and TIR/Хранилище каналов'
DATA_PATH = 'Data'
cur_file = 'hsi.npy'

y = yadisk.YaDisk(token=TOKEN)

parser = Parser()
nm = parser.wave

hsi = None
stat = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/stat')
def get_stat():
    global stat
    stat = Statistics(hsi)
    return stat.stat_dct

@app.get('/stat/regression')
def get_regression(b1: int, b2: int):
    return stat.regression(b1, b2)
    

@app.get('/bands/{expr}')
def get_channel(expr: str):
    parser(expr)
    channel = parser.channel(hsi)

    return {'x': channel.tolist()}

@app.get('/spectre')
def get_spectre(x: int, y: int):
    spectre = hsi[y, x, :]

    global stat
    if stat is None: stat = Statistics(hsi)

    res_stat = stat.diff(y, x)

    return {'spectre': spectre.tolist(), 'nm': nm.tolist()} | res_stat


from algorithms.classification.sam import SAM
from algorithms.classification.sid import SID
from algorithms.classification.sca import SCA

@app.get('/classification')
def classification(algorithm: str, x: int, y: int):
    algorithms = {'sam': SAM, 'sid': SID, 'sca': SCA}
    angle_maps = algorithms[algorithm](hsi, [hsi[y, x]])
    print(angle_maps[0])

    return {'x': angle_maps[0].tolist()}

@app.get('/rgb')
def get_rgb():
    rgb = np.around(hsi[:, :, (70, 51, 18)] * 255).astype(int)
    return {'x' : rgb.tolist()}

from sklearn.cluster import KMeans, OPTICS, AffinityPropagation, Birch, SpectralClustering, MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

@app.get('/clusterization')
def get_clusters(k: int, method: str):
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

    # hsi_table = hsi[200:300, 200:300].reshape(-1, 204)
    hsi_table = hsi.reshape(-1, 204)
    results = model.fit_predict(hsi_table).reshape(hsi.shape[0], hsi.shape[1])
    
    return {'classes': results.tolist()}

from pysptools.abundance_maps.amaps import FCLS, UCLS, NNLS
from pysptools.eea import NFINDR, PPI

@app.get('/extract')
def get_extract(method: str, amap: str, k: int):
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
    
@app.get('/download')
def download_file(path: str):
    download_from_path(path)
    global cur_file
    cur_file = path.split('/')[-1]

    global hsi
    path_file = f'{DATA_PATH}/{cur_file}'
    hsi = loader(file=path_file)

    return {'result': list(hsi.shape), 'file': cur_file}