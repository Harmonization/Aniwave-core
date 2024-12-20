import numpy as np

from hsip.processing.processing import normalize, sigma_maximum_filter

def hsi_synthesize_rgb(spectral_data: np.ndarray, rgb_bands: list | np.ndarray = None, wavelengths: list | np.ndarray = None):
    """
    Synthesizes an RGB image from hyperspectral data.

    Parameters
    ----------
    spectral_data : np.ndarray
        Hyperspectral image data, expected to be a 3D array of shape (height, width, bands).
    rgb_bands : list of int, optional
        A list containing the band indices for red, green, and blue channels, in that order.
        Must be of length 3. If provided, `wavelengths` is ignored.
    wavelengths : list of float, optional
        A list containing the wavelengths corresponding to each band in `spectral_data`.
        If provided, the closest wavelengths to 650 nm (red), 550 nm (green), and 450 nm (blue) are used.

    Returns
    -------
    np.ndarray
        A 3D array of shape (height, width, 3) representing the synthesized RGB image.

    Examples
    --------
    Using band indices directly:
    >>> from hsip.rgb.rgb import hsi_synthesize_rgb
    >>> spectral_data = np.random.rand(100, 100, 224)  # Example hyperspectral data
    >>> rgb_bands = [50, 100, 150]  # Example red, green, blue bands
    >>> rgb_image = synthesize_rgb(spectral_data, rgb_bands=rgb_bands)
    >>> print(rgb_image.shape)
    (100, 100, 3)

    Using wavelengths:
    >>> wavelengths = np.linspace(400, 700, 224)  # Example wavelength data
    >>> rgb_image = synthesize_rgb(spectral_data, wavelengths=wavelengths)
    >>> print(rgb_image.shape)
    (100, 100, 3)
    """
    if rgb_bands is None and wavelengths is None:
        raise ValueError("Either `rgb_bands` or `wavelengths` must be provided.")

    if rgb_bands is not None:
        if isinstance(rgb_bands, np.ndarray):
            rgb_bands = rgb_bands.tolist()

        if len(rgb_bands) != 3:
            raise ValueError("`rgb_bands` must contain exactly three indices for red, green, and blue bands.")
        red_idx, green_idx, blue_idx = rgb_bands
        
    elif wavelengths is not None:
        if isinstance(wavelengths, np.ndarray):
            wavelengths = wavelengths.tolist()
        
        if len(wavelengths) != spectral_data.shape[-1]:
            raise ValueError("The length of `wavelengths` must match the number of bands in `spectral_data`.")
        red_idx = np.argmin(np.abs(np.array(wavelengths) - 650))  # Closest to 650 nm (red)
        green_idx = np.argmin(np.abs(np.array(wavelengths) - 550))  # Closest to 550 nm (green)
        blue_idx = np.argmin(np.abs(np.array(wavelengths) - 450))  # Closest to 450 nm (blue)
    else:
        raise ValueError("Invalid parameters: provide either `rgb_bands` or `wavelengths`.")

    red_band = spectral_data[..., red_idx]
    red_band = normalize(red_band) * 255
    
    green_band = spectral_data[..., green_idx]
    green_band = normalize(green_band) * 255
    
    blue_band = spectral_data[..., blue_idx]
    blue_band = normalize(blue_band) * 255
    
    rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)
    rgb_image = np.uint8(rgb_image)

    return rgb_image

def simple_synthesize_rgb(band_data: list, sig_max_filt: float = None):
    '''
    Generate an RGB image from three hyperspectral bands.

    This function synthesizes a simple RGB image from three provided hyperspectral bands,
    corresponding to the red, green, and blue channels. Optionally, it applies a sigma 
    maximum filter to each band for noise reduction before normalization and scaling.

    Parameters
    ----------
    band_data : list of np.ndarray
        A list containing exactly three 2D arrays, each representing one band of hyperspectral 
        data for the red, green, and blue channels. Each array must have the same shape.
    sig_max_filt : float, optional
        A sigma value to apply the `sigma_maximum_filter` function to each band. If provided, 
        it is used for noise reduction before synthesizing the RGB image.

    Returns
    -------
    rgb_image : np.ndarray
        A 3D array of shape `(height, width, 3)` representing the synthesized RGB image.
        The output is of type `np.uint8` with pixel values scaled to the range [0, 255].

    Examples
    --------
    Generate an RGB image with sigma maximum filtering
    >>> import numpy as np
    >>> from hsip.rgb.rgb import simple_synthesize_rgb
    >>> band_red = np.random.rand(100, 100)
    >>> band_green = np.random.rand(100, 100)
    >>> band_blue = np.random.rand(100, 100)
    >>> band_data = [band_red, band_green, band_blue]
    >>> rgb_image = simple_synthesize_rgb(band_data, sig_max_filt=3)
    >>> print(rgb_image.shape)
    (100, 100, 3)
    '''
    if len(band_data) != 3:
        raise ValueError("`band_data` must contain exactly three bands for red, green, and blue bands.")
    
    if sig_max_filt is not None:
        band_data = [sigma_maximum_filter(band[..., np.newaxis], sigma=3)[..., 0] for band in band_data]
        
    band_data = [normalize(band) * 255 for band in band_data]
    
    rgb_image = np.stack(band_data, axis=-1)
    rgb_image = np.uint8(rgb_image)

    return rgb_image