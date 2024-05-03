import numpy as np

def SCA(hsi, U):
    rows, cols, bands = hsi.shape
    hsi_table = hsi.reshape(rows*cols,bands)
    result = []
    for u in U:
        mean_x = np.mean(hsi_table, axis=1) # вектор n
        mean_y = np.mean(u) # скаляр
        mean_xy = np.mean(hsi_table * u, axis=1) # вектор n
        sigma_x = np.std(hsi_table, axis=1) # вектор n
        sigma_y = np.std(u) # скаляр
        sca = (mean_xy - mean_x * mean_y) / (sigma_x * sigma_y)
        sca[np.isnan(sca)] = 0

        result.append(sca.reshape(rows, cols))
    return result