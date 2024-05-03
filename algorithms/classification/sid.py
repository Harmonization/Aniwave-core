import numpy as np

def SID(hsi, U):
    rows, cols, bands = hsi.shape
    hsi_table = hsi.reshape(rows*cols,bands)
    result = []
    for u in U:
        mod_u = np.linalg.norm(u) # норма вектора
        mod_hsi = np.sqrt(np.sum(hsi_table**2, axis=1)) 
        norm_u = u / mod_u # нормализованный вектор
        norm_hsi = hsi_table / mod_hsi[:, np.newaxis]
        sid = np.sum(norm_hsi * np.log(norm_hsi / norm_u) + norm_u * np.log(norm_u / norm_hsi), axis=1)
        sid[np.isnan(sid)] = 0

        result.append(sid.reshape(rows, cols))
    return result