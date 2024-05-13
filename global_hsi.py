import os

import numpy as np
import yadisk

from loader import loader, download_from_path
from parse import Parser

class HSI:
    DATA_PATH = 'Data'

    def __init__(self) -> None:
        self.parser = Parser()
        self.y = yadisk.YaDisk(token='y0_AgAAAAAsGvHOAApsKgAAAADrhnQx1pqpt1zjQPCSV28z1eLp1OYhH6g')

    def get(self):
        return self.hsi
    
    def shape(self):
        return self.hsi.shape
    
    def get_nm(self):
        return self.parser.wave
    
    def get_index(self, expr: str):
        self.parser(expr)
        channel = self.parser.channel(self.get())
        return {'spectral_index': channel.tolist()}
    
    def get_rgb(self):
        rgb = np.flip(np.around(self[70, 51, 18] * 255).astype(int), axis=0)
        return {'rgb' : rgb.tolist()}

    def download(self, path: str):
        filename = path.split('/')[-1]
        path_file = f'{self.DATA_PATH}/{filename}'
        
        if os.path.exists(path_file):
            # файл уже скачан
            self.hsi = np.load(path_file)
            return {'name': filename}
        
        # иначе загружаем
        self.y.download(path, path_file)
        return {'name': filename}

    
    def __getitem__(self, ij):
        if len(ij) == 2:
            i, j = ij
            return self.hsi[i, j]
        else:
            return self.hsi[:, :, ij]
        
    def diff(self, i, j):
        spectre = self[i, j]
        dx = 1 / self.shape()[2]
        dy_x = np.diff(spectre) / dx
        return {'derivative': np.around(dy_x, 3).tolist(),
                
                'max_spectre': round(float(spectre.max()), 3), 
                'min_spectre': round(float(spectre.min()), 3), 
                'mean_spectre': round(float(spectre.mean()), 3), 
                'std_spectre': round(float(spectre.std()), 3),
                'scope_spectre': round(float(spectre.max() - spectre.min()), 3),
                'iqr_spectre': round(np.subtract(*np.percentile(spectre, [75, 25])), 3),
                'q1_spectre': round(float(np.percentile(spectre, 25)), 3),
                'median_spectre': round(float(np.median(spectre)), 3),
                'q3_spectre': round(float(np.percentile(spectre, 75)), 3),
                
                'max_deriv': round(float(dy_x.max()), 3), 
                'min_deriv': round(float(dy_x.min()), 3), 
                'mean_deriv': round(float(dy_x.mean()), 3), 
                'std_deriv': round(float(dy_x.std()), 3),
                'scope_deriv': round(float(dy_x.max() - dy_x.min()), 3),
                'iqr_deriv': round(np.subtract(*np.percentile(dy_x, [75, 25])), 3),
                'q1_deriv': round(float(np.percentile(dy_x, 25)), 3),
                'median_deriv': round(float(np.median(dy_x)), 3),
                'q3_deriv': round(float(np.percentile(dy_x, 75)), 3)}