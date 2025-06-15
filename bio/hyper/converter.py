from os import listdir
from tkinter import Tk, filedialog

import numpy as np
import spectral.io.envi as envi
from hsip.reader import open_TIF

def open_explorer():
    window = Tk()
    window.geometry("5x5")
    path = filedialog.askopenfilename(initialdir = "/", title = "Выберите HSI",
                                          filetypes = ((".hdr, .tif, .npy", "*.hdr* *.tif* *.npy*"), ("Все файлы", "*.*")))
    
    window.destroy()
    return path

def convert_hsi(path: str):
    # Открыть HSI и конвертировать в формат np.array
    
    _, filetype = path.split('.')
    nm = np.array([]).astype(int)
    rgb_bands = None

    if filetype == 'hdr':
        img = envi.open(path)
        # arr = np.array(img.load())
        arr = img.load()
        hsi = np.rot90(arr, k=1)
        
        if 'wavelength' in img.metadata: 
            waves = list(map(float, img.metadata['wavelength']))
            nm = np.around(waves).astype(int)

        if 'default bands' in img.metadata:
            rgb_bands = list(map(int, img.metadata['default bands']))
    
    elif filetype == 'npy':
        hsi = np.load(path, mmap_mode='r')
        
        folder = '/'.join(path.split('/')[:-1])
        if 'nm.npy' in listdir(folder):
            nm = np.load(f'{folder}/nm.npy')

    elif filetype == 'tif':
        hsi = open_TIF(path).transpose(2,1,0)

    return hsi, nm, rgb_bands