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

def save_explorer(name: str = ''):
    window = Tk()
    window.geometry("5x5")
    # path = filedialog.askopenfilename(initialdir = "/", title = "Выберите HSI",
    #                                       filetypes = ((".xlsx", "*.xlsx*"), ("Все файлы", "*.*")))

    # path = filedialog.askdirectory(initialdir = "/", title = "Сохранить")
    path = filedialog.asksaveasfilename(initialdir = "/", title = "Сохранить", filetypes = ((".xlsx", "*.xlsx*"), ("Все файлы", "*.*")),initialfile=name)
    window.deiconify()
    window.destroy()
    return path + '.xlsx'

def convert_hsi(path: str):
    # Открыть HSI и конвертировать в формат np.array
    print('||||||||||||||||||||||||||||||||||')
    print(path)
    print('||||||||||||||||||||||||||||||||||')

    _, filetype = path.split('.')
    nm = np.array([]).astype(int)
    rgb_bands = None

    if filetype == 'hdr':
        img = envi.open(path)
        # arr = np.array(img.load())
        hsi = img.load()

        path_elements = path.split('/')
        FOLDER = '/'.join(path_elements[:-1])
        FILENAME, FORMAT = path_elements[-1].split('.')
        DARKREF = f'DARKREF_{FILENAME}.{FORMAT}'
        WHITEREF = f'WHITEREF_{FILENAME}.{FORMAT}'
        FILES = listdir(FOLDER)
        CONDITION = f'{FILENAME}.raw' in FILES and DARKREF in FILES and WHITEREF in FILES 
        
        if CONDITION:
            # Открытие белых и черных стандартов
            dark = envi.open(f'{FOLDER}/{DARKREF}').load()
            white = envi.open(f'{FOLDER}/{WHITEREF}').load()

            # Усреднение по строчке
            dark = np.mean(dark, axis=0)[np.newaxis, ...]
            white = np.mean(white, axis=0)[np.newaxis, ...]

            hsi = (hsi - dark) / (white - dark)

            del dark
            del white

        hsi = np.rot90(hsi, k=3)
        
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

    return hsi, nm, waves, rgb_bands