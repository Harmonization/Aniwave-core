import pandas as pd

def open_tir(path: str):
    '''
    Открытие термального изображения по выбранному пути. 
    Изображение открывается из табличного формата xlsx.

    Параметры
    ----------
    hpathsi : str
        Путь к таблице с данными.
    
    Возвращает
    -------
    np.ndarray
        2D массив numpy размера (n_rows, n_columns) содержащий Thermal Infrared изображение.
    
    Примеры
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from bio.thermal import open_tir
    >>> result = open_tir('folder_1/tir_5.xlsx')
    >>> plt.imshow(result)
    >>> plt.show()
    >>> print(result.shape)
    (320, 240)  # Example output
    '''
    tir = pd.read_excel(path, header = None).to_numpy()
    return tir