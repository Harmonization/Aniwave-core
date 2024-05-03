import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import entropy

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