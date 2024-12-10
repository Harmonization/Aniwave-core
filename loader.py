import os

import numpy as np
import spectral.io.envi as envi
from hsip.reader import open_TIF
import yadisk

DATA_PATH = 'Data'
MAIN_ROOT = 'Datasets and Program/HSI and TIR/Aniwave'
TOKEN = 'y0_AgAAAAAsGvHOAApsKgAAAADrhnQx1pqpt1zjQPCSV28z1eLp1OYhH6g'
y = yadisk.YaDisk(token=TOKEN)

def download(path: str):
    filename = path.split('/')[-1]
    path_file = f'{DATA_PATH}/{filename}'
    
    if not os.path.exists(path_file):
        # если файл не скачан
        y.download(f'{MAIN_ROOT}/{path}', path_file)
    
    hsi = np.load(path_file)
    return hsi

def open(name: str):
    # Открыть HSI
    
    filename, filetype = name.split('.')
    if filetype == 'hdr':
        path = f'Data/{filename}.hdr'
        img = envi.open(path)
        arr = np.array(img.load())
        hsi = np.rot90(arr, k=1)
    
    elif filetype == 'npy':
        path = f'Data/{filename}.npy'
        hsi = np.load(path)

    elif filetype == 'tif':
        path = f'Data/{filename}.tif'
        hsi = open_TIF(path).transpose(2,1,0)

    return hsi
    
def upload(data: np.ndarray, filename: str):
    np.save(filename, data)
    y.upload(filename, f'{MAIN_ROOT}/{filename}')
    os.remove(filename)

# def upload(self, method: str, h: int, expr: str, t1: float, t2: float, rois: list[list[int]], filename: str):
#     # Загрузить на диск
#     print(method, h, expr, t1, t2, rois, filename)
    
#     static_name = 'Data/roi.npy' # имя временного сохранения на ПК

#     # ROI
#     print('Радзеление на ROI')
#     hsi_rois = [self.get()[y0: y1, x0: x1].copy() for (x0, x1, y0, y1) in rois]
#     print('Разделилось на ROI')
    
#     for i, hsi_roi in enumerate(hsi_rois):
#         # Сглаживание
#         print('Сглаживание')

#         filter_func = self.hsi_filters[method]
#         hsi_roi_smooth = filter_func(hsi_roi, h)

#         # Маска
#         print('маск')

#         channel = str2indx(hsi_roi_smooth, expr)
#         hsi_roi_smooth[channel < t1] = 0
#         hsi_roi_smooth[channel > t2] = 0

#         print('маск')

#         # Сохранение
#         print('сейв')

#         name_i = f'{filename}_{i}.npy' # имя файла на диске
#         np.save(static_name, hsi_roi_smooth)
#         y.upload(static_name, f'{MAIN_ROOT}/{name_i}')

#         print('сейв')
#         print(i)

    
#     os.remove(static_name)
