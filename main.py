from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from global_hsi import HSI
import algorithms.statistics.stat as STAT
from get_class import get_classification, get_clusterization, get_extraction

hsi = HSI()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/download')
def download_file(path: str):
    return hsi.download(path) | hsi.get_rgb()

@app.get('/spectre')
def get_spectre(x: int, y: int):
    spectre = hsi[y, x]
    diff = hsi.diff(y, x)

    return {'spectre': spectre.tolist()} | diff

@app.get('/bands/{expr}')
def get_channel(expr: str):
    return hsi.get_index(expr)

@app.get('/stat/regression')
def get_regression(b1: int, b2: int):
    return STAT.get_regression(hsi.get(), b1, b2)

@app.get('/stat/{stat_name}')
def get_stat(stat_name: str, mode: str):
    if stat_name == 'matrix_correlation':
        return STAT.correlation_matrix(hsi.get())
    else:
        return STAT.get_stat_hsi(hsi.get(), stat_name, mode)

@app.get('/rgb')
def get_rgb():
    return hsi.get_rgb()

@app.get('/classification')
def classification(algorithm: str, x: int, y: int):
    return get_classification(hsi.get(), algorithm, x, y)

@app.get('/clusterization')
def get_clusters(k: int, method: str):
    return get_clusterization(hsi.get(), k, method)

@app.get('/extract')
def get_extract(method: str, amap: str, k: int):
    return get_extraction(hsi.get(), method, amap, k)