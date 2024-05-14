import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy
from functools import partial

def mean_matrix(spectre: np.ndarray[float]) -> np.ndarray[float]:
    x, y = np.meshgrid(spectre, spectre)
    return np.fromfunction(lambda i, j: (x-y) / (x+y), (204, 204))

def correlation_matrix(hsi):
    hsi_table = hsi.reshape(hsi.shape[0] * hsi.shape[1], hsi.shape[2]).T # shape = (204, N)
    return np.around(np.corrcoef(hsi_table), 3).tolist()

def get_stat_value(data, stat_func, mode: str = 'hsi', round_n=3):
    # моды: hsi, matrix и поканальный
    # data_without_null = data[data != 0]
    # mask = (data != 0) & (~np.isnan(data))
    match mode:
        case 'hsi':
            return round(float(stat_func(data)), round_n)
        case 'matrix':
            # Находим поканальный массив признаков, и на его основе формируем матрицу
            stat_arr = np.around(np.array(stat_func(data, axis=(0, 1))).astype(float), round_n)
            mx = mean_matrix(stat_arr)
            mx[np.isnan(mx)] = 0
            return mx.tolist()
        case _:
            return np.around(np.array(stat_func(data, axis=(0, 1))).astype(float), round_n).tolist()

def get_stat_hsi(hsi, name: str, mode: str = 'hsi'):

    def get_entropy(hsi, axis=None):
        if axis is None:
            mask = ~np.isnan(hsi) & (hsi != 0)
            return entropy(np.unique(hsi[mask], return_counts=True)[1])
        else:
            res = []
            for band in range(hsi.shape[-1]):
                channel = hsi[..., band]
                mask = ~np.isnan(channel) & (channel != 0)
                uniq = np.unique(channel[mask], return_counts=True)[1]
                res.append(entropy(uniq))
            return res
            # return [entropy(np.unique(hsi[..., band], return_counts=True)[1]) for band in range(hsi.shape[-1])]

    def get_max(data: np.ndarray, axis: tuple | None = None):
        mask = ~np.isnan(data) & (data != 0)
        return data.max(axis=axis, initial=-10, where=mask)
    
    def get_min(data: np.ndarray, axis: tuple | None = None):
        mask = ~np.isnan(data) & (data != 0)
        return data.min(axis=axis, initial=10, where=mask)
    
    def get_mean(data: np.ndarray, axis: tuple | None = None):
        mask = ~np.isnan(data) & (data != 0)
        return data.mean(axis=axis, where=mask)
    
    def get_std(data: np.ndarray, axis: tuple | None = None):
        mask = ~np.isnan(data) & (data != 0)
        return data.std(axis=axis, where=mask)
    
    def get_scope(data: np.ndarray, axis: tuple | None = None):
        return get_max(data, axis) - get_min(data, axis)
    
    def get_median(data: np.ndarray, axis: tuple | None = None):
        mask = ~np.isnan(data) & (data != 0)
        data_without_zero = np.ma.array(data, mask=~mask)
        return np.ma.median(data_without_zero, axis=axis)
    
    def get_q1(data: np.ndarray, axis: tuple | None = None):
        mask = ~np.isnan(data) & (data != 0)
        data_without_zero = data.copy()
        data_without_zero[~mask] = np.nan
        return np.nanpercentile(data_without_zero, q=25, axis=axis)
    
    def get_q3(data: np.ndarray, axis: tuple | None = None):
        mask = ~np.isnan(data) & (data != 0)
        data_without_zero = data.copy()
        data_without_zero[~mask] = np.nan
        return np.nanpercentile(data_without_zero, q=75, axis=axis)
    
    def get_iqr(data: np.ndarray, axis: tuple | None = None):
        return get_q3(data, axis) - get_q1(data, axis)

    stat_functions = {
        'max': get_max, 
        'min': get_min,
        'mean': get_mean, 
        'std': get_std, 
        'scope': get_scope,
        'iqr': get_iqr,
        'entropy': get_entropy,
        'q1': get_q1,
        'median': get_median,
        'q3': get_q3
    }
    
    return get_stat_value(hsi, stat_functions[name], mode)

def get_regression(hsi, b1: int, b2: int):
    model = LinearRegression()
    band_1, band_2 = hsi[:, :, b1], hsi[:, :, b2]
    model.fit(band_1.reshape(-1, 1), band_2.reshape(-1, 1))

    arr_1, arr_2 = band_1.ravel(), band_2.ravel()
    a, b = model.coef_[0], model.intercept_
    x = np.linspace(0, band_1.max(), 10)
    y = a * x + b
    r = np.corrcoef(arr_1, arr_2)[0, 1]
    R = r**2
    mean_elastic = a * band_1.mean() / band_2.mean()
    beta = a * band_1.std() / band_2.std()
    indx = np.random.choice(len(arr_1), len(arr_1)//10)
    points_1 = np.around(arr_1[indx], 3).tolist()
    points_2 = np.around(arr_2[indx], 3).tolist()
    return {'b1': b1, 
            'b2': b2, 
            'x': x.tolist(), 
            'y': y.tolist(), 
            'a': round(float(a), 3), 
            'b': round(float(b)),
            'points_1': points_1,
            'points_2': points_2,
            'correlation': round(float(r), 3), 
            'determination': round(float(R), 3), 
            'elastic': round(float(mean_elastic), 3), 
            'beta': round(float(beta), 3)}

class Statistics:

    def __init__(self, hsi):
        self.hsi = hsi

        hsi_table = hsi.reshape(hsi.shape[0] * hsi.shape[1], hsi.shape[2]).T # shape = (204, N)
        self.stat_dct = {
            'max_hsi': round(float(hsi.max()), 3),
            'min_hsi': round(float(hsi.min()), 3),
            'mean_hsi': round(float(hsi.mean()), 3),
            'std_hsi': round(float(hsi.std()), 3),
            'scope_hsi': round(float(hsi.max() - hsi.min()), 3),
            'iqr_hsi': round(np.subtract(*np.percentile(hsi, [75, 25])), 3),
            'entropy_hsi': round(entropy(np.unique(self.hsi, return_counts=True)[1]), 3),
            'q1_hsi': round(float(np.percentile(hsi, 25)), 3),
            'median_hsi': round(float(np.median(hsi)), 3),
            'q3_hsi': round(float(np.percentile(hsi, 75)), 3),

            'max_bands': [],
            'max_bands': [],
            'min_bands': [],
            'mean_bands': [],
            'std_bands': [],
            'scope_bands': [],
            'iqr_bands': [],
            'entropy_bands': [],
            'q1_bands': [],
            'median_bands': [],
            'q3_bands': [],

            'hsi_matrix_correlation': np.around(np.corrcoef(hsi_table), 3).tolist()
        }
        for b in range(hsi.shape[2]):
            band = hsi[:, :, b]
            self.stat_dct['max_bands'].append(round(float(band.max()), 3))
            self.stat_dct['min_bands'].append(round(float(band.min()), 3))
            self.stat_dct['mean_bands'].append(round(float(band.mean()), 3))
            self.stat_dct['std_bands'].append(round(float(band.std()), 3))
            self.stat_dct['scope_bands'].append(round(float(band.max() - band.min()), 3))
            self.stat_dct['iqr_bands'].append(round(float(np.subtract(*np.percentile(band, [75, 25]))), 3))
            self.stat_dct['entropy_bands'].append(round(entropy(np.unique(band, return_counts=True)[1]), 3))
            self.stat_dct['q1_bands'].append(round(float(np.percentile(band, 25)), 3))
            self.stat_dct['median_bands'].append(round(float(np.median(band)), 3))
            self.stat_dct['q3_bands'].append(round(float(np.percentile(band, 75)), 3))

    def diff(self, i, j):
        spectre = self.hsi[i, j, :]
        dx = 1 / self.hsi.shape[2]
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
    
    def regression(self, b1, b2):
        model = LinearRegression()
        band_1, band_2 = self.hsi[:, :, b1], self.hsi[:, :, b2]
        model.fit(band_1.reshape(-1, 1), band_2.reshape(-1, 1))

        arr_1, arr_2 = band_1.ravel(), band_2.ravel()
        a, b = model.coef_[0], model.intercept_
        x = np.linspace(0, band_1.max(), 10)
        y = a * x + b
        r = np.corrcoef(arr_1, arr_2)[0, 1]
        R = r**2
        mean_elastic = a * band_1.mean() / band_2.mean()
        beta = a * band_1.std() / band_2.std()
        indx = np.random.choice(len(arr_1), len(arr_1)//10)
        points_1 = np.around(arr_1[indx], 3).tolist()
        points_2 = np.around(arr_2[indx], 3).tolist()
        return {'b1': b1, 
                'b2': b2, 
                'x': x.tolist(), 
                'y': y.tolist(), 
                'a': round(float(a), 3), 
                'b': round(float(b)),
                'points_1': points_1,
                'points_2': points_2,
                'correlation': round(float(r), 3), 
                'determination': round(float(R), 3), 
                'elastic': round(float(mean_elastic), 3), 
                'beta': round(float(beta), 3)}