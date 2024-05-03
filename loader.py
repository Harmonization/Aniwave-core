import os

import numpy as np
import yadisk

MAIN_ROOT = 'Datasets and Program/HSI and TIR/Хранилище каналов'
SERVER_ROOT = 'Data'

y = yadisk.YaDisk(token='y0_AgAAAAAsGvHOAApsKgAAAADrhnQx1pqpt1zjQPCSV28z1eLp1OYhH6g')

def download_from_path(path):
    filename = path.split('/')[-1]
    save_path = f'{SERVER_ROOT}/{filename}'
    if os.path.exists(save_path): 
        return 'файл уже скачан'
    
    y.download(path, save_path)

    return 'файл скачался'

def loader(file: str):
    return np.load(file)

