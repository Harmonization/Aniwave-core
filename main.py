import json, os

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

# @app.get('/settings')
# def settings():
#     with open('settings.json', 'r', encoding='utf-8') as settings_json:
#         settings = json.load(settings_json)

#     return settings

@app.get('/convert')
def get_convert():
    return getter.get_convert()

@app.get('/save_convert')
def save_convert():
    return getter.save_convert()

@app.get('/save_band')
def save_band(expr: str = ''):
    return getter.save_band(expr)

@app.get('/files')
def get_files(filetypes: list[str] = ['npy', 'hdr', 'tif'], path: str = 'Data'):
    # Получить все загруженные файлы выбранных типов
    if not os.path.isdir(path):
        # Если директория не существует, создаем
        os.makedirs(path)
    
    files = list(filter(lambda el: el.split('.')[-1] in filetypes, os.listdir(path)))
    return {'downloadedFiles': files}

@app.get('/change_channels')
def change_channels(channels_expr: str = '0-203'):
    return getter.change_channels(channels_expr)

@app.get('/change_roi')
def change_roi(x0: int = 0, x1: int = 0, y0: int = 0, y1: int = 0):
    return getter.change_roi(x0, x1, y0, y1)

@app.get('/change_thr')
def change_thr(thr_expr: str, lower: float, upper: float):
    return getter.change_thr(thr_expr, lower, upper)

@app.get('/open')
def open_image(name: str):
    return getter.open_hsi(name) #| getter.get_rgb()

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
def get_sign(x: int, y: int, method: str = '', h: int = 5):
    return getter.get_signal(y, x, method, h)

@app.get('/bands')
def get_channel(expr: str):
    # path_file = 'img1.png'
    # return FileResponse(path_file)
    return getter.get_indx(expr)

@app.get('/bands_png')
def get_channel_png(expr: str, t: float = 0, condition: str = ''):
    getter.get_indx(expr, t, condition)
    path_file = 'Data/band_0.png'
    return FileResponse(path_file)

@app.get('/rgb')
def get_tir():
    getter.get_rgb()
    path_file = 'Data/rgb.png'
    return FileResponse(path_file)

@app.get('/hist')
def get_hist():
    return getter.get_hist()

@app.get('/idx_mx')
def get_idx_mx(name: str):
    getter.get_idx_mx(name)
    path_file = 'Data/mx_0.png'
    return FileResponse(path_file)

@app.get('/regression')
def get_regression(b1: int, b2: int):
    return getter.get_regression(b1, b2)

@app.get('/corr_mx')
def get_corr_mx(startBand: int = 0, endBand: int = 203):
    return getter.get_corr_mx(startBand, endBand)

@app.get('/rgb')
def get_rgb():
    return getter.get_rgb()

@app.get('/classes')
def get_spectral_classes(method: str, x: int, y: int):
    return getter.get_spectral_classes(method, x, y)

@app.get('/clusters')
def get_clusters(k: int, method: str):
    res = getter.get_clusters(k, method)
    path_file = 'Data/segmentation.png'
    return FileResponse(path_file)

@app.get('/clusters_2')
def get_clusters_2(thr: float = .99, method: str = 'cosine', metrics: str = 'cosine'):
    return getter.get_clusters_2(thr, method, metrics)

@app.get('/clusters_corr')
def get_cluster_corr( mode: str = 'centroids'):
    return getter.get_cluster_corr(mode)

@app.get('/reley')
def get_reley():
    return getter.get_reley()

@app.get('/sigma')
def get_reley(sigma: int = 2):
    return getter.get_sigma(sigma)

@app.get('/rgb_synthesize')
def get_rgb_synthesize( red: int = 70, green: int = 51, blue: int = 19, red_mode: int | None = None, green_mode: int | None = None, blue_mode: int | None = None):
    return getter.get_rgb_synthesize(red, green, blue, red_mode, green_mode, blue_mode)

@app.get('/emd')
def get_emd(number_of_modes: int = 8, windows_size: list[int] = [3, 3, 5]):
    return getter.get_emd(number_of_modes, windows_size)

@app.get('/emd_channel')
def get_emd(n_mode: int = 0, n_band: int = 0):
    return getter.get_emd_channel(n_mode, n_band)

@app.get('/endmembers')
def get_endmembers(method: str, k: int):
    return getter.get_endmembers(method, k)

@app.get('/amaps')
def get_amaps(method: str, endmembers: list = Query()):
    return getter.get_amaps(method, endmembers)

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

@app.get('/channel_story')
def get_channel_story():
    return getter.get_channel_story()

if __name__ == '__main__':
    import uvicorn
    import webbrowser
    webbrowser.open('https://harmonization.github.io/Aniwave/')
    uvicorn.run(app, host='localhost', port=8000)