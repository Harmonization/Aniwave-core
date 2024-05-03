import numpy as np

def SAM(hsi, U):
    rows, cols, bands = hsi.shape
    hsi_table = hsi.reshape(rows*cols,bands)
    result = []
    for u in U:
        mod_u = np.linalg.norm(u)
        mod_hsi = np.sqrt(np.sum(hsi_table**2, axis=1))
        sam = (hsi_table @ u) / (mod_hsi * mod_u)
        sam[np.isnan(sam)] = 1

        result.append(np.arccos(sam).reshape(rows, cols))
    # result.append(np.argmax(result, axis=2))
    return result