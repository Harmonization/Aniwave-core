import json, os

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import getter

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def exist_server():
    return

@app.get('/settings')
def settings():
    with open('Data/Settings/settings.json', 'r', encoding='utf-8') as settings_json:
        settings = json.load(settings_json)

    return settings

@app.get('/files')
def get_files(filetypes: list[str] = ['npy', 'hdr', 'tif']):
    # Получить все загруженные файлы выбранных типов
    files = [el for el in os.listdir('Data') if len(el.split('.')) > 1 and el.split('.')[-1] in filetypes]
    return {'downloadedFiles': files}

@app.get('/open')
def open_image(name: str):
    return getter.open_hsi(name) | getter.get_rgb(name)

@app.get('/tir')
def get_tir(name: str = '3.xlsx'):
    return getter.get_tir(name)

class DownloadOptions(BaseModel):
    paths: list[str]

@app.post('/download_files')
def download_file(options: DownloadOptions):
    paths = options.paths
    for path in paths:
        getter.download_hsi(path)
    return

@app.get('/signal')
def get_sign(name_hsi: str, x: int, y: int, method: str = '', h: int = 5):
    return getter.get_signal(name_hsi, y, x, method, h)

@app.get('/bands')
def get_channel(name_hsi: str, expr: str):
    return getter.get_indx(name_hsi, expr)

@app.get('/idx_mx')
def get_idx_mx(name_hsi: str, name: str, startBand: int = 0, endBand: int = 203):
    return getter.get_idx_mx(name_hsi, name, startBand, endBand)

@app.get('/regression')
def get_regression(name_hsi: str, b1: int, b2: int):
    return getter.get_regression(name_hsi, b1, b2)

@app.get('/corr_mx')
def get_corr_mx(name_hsi: str, startBand: int = 0, endBand: int = 203):
    return getter.get_corr_mx(name_hsi, startBand, endBand)

@app.get('/rgb')
def get_rgb(name_hsi: str):
    return getter.get_rgb(name_hsi)

@app.get('/classes')
def get_spectral_classes(name_hsi: str, method: str, x: int, y: int):
    return getter.get_spectral_classes(name_hsi, method, x, y)

@app.get('/clusters')
def get_clusters(name_hsi: str, k: int, method: str):
    return getter.get_clusters(name_hsi, k, method)

@app.get('/clusters_2')
def get_clusters_2(name_hsi: str, thr: float = .99, method: str = 'cosine', metrics: str = 'cosine'):
    return getter.get_clusters_2(name_hsi, thr, method, metrics)

@app.get('/clusters_corr')
def get_cluster_corr(name_hsi: str, mode: str = 'centroids'):
    return getter.get_cluster_corr(name_hsi, mode)

@app.get('/reley')
def get_reley(name_hsi: str):
    return getter.get_reley(name_hsi)

@app.get('/sigma')
def get_reley(name_hsi: str, sigma: int = 2):
    return getter.get_sigma(name_hsi, sigma)

@app.get('/rgb_synthesize')
def get_rgb_synthesize(name_hsi: str, red: int = 70, green: int = 51, blue: int = 19, red_mode: int | None = None, green_mode: int | None = None, blue_mode: int | None = None):
    return getter.get_rgb_synthesize(name_hsi, red, green, blue, red_mode, green_mode, blue_mode)

@app.get('/emd')
def get_emd(name_hsi: str, number_of_modes: int = 8, windows_size: list[int] = [3, 3, 5]):
    return getter.get_emd(name_hsi, number_of_modes, windows_size)

@app.get('/emd_channel')
def get_emd(name_hsi: str, n_mode: int = 0, n_band: int = 0):
    return getter.get_emd_channel(name_hsi, n_mode, n_band)

@app.get('/endmembers')
def get_endmembers(name_hsi: str, method: str, k: int):
    return getter.get_endmembers(name_hsi, method, k)

@app.get('/amaps')
def get_amaps(name_hsi: str, method: str, endmembers: list = Query()):
    return getter.get_amaps(name_hsi, method, endmembers)

class Settings(BaseModel):
    channel: str
    index_story: list[str]
    rois_story: list[str]
    colormap: str
    filter: str
    h: int
    t1: float
    t2: float

@app.post('/settings/save')
def save_settings(settings: Settings):
    try:
        with open('Data/Settings/settings.json', 'w', encoding='utf-8') as settings_json:
            json.dump({
                'channel': settings.channel,
                'index_story': settings.index_story,
                'rois_story': settings.rois_story,
                'colormap': settings.colormap,
                'filter': settings.filter,
                'h': settings.h,
                't1': settings.t1,
                't2': settings.t2
            }, settings_json, ensure_ascii=False, indent=4)
            
        result = 'Успешно загружено'
    except:
        result = 'Ошибка загрузки'
    return {'result': result}