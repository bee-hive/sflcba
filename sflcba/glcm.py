# sflcba/glcm.py
import numpy as np
from scipy.signal import convolve2d
from skimage.transform import integral_image
from skimage.feature import graycomatrix, graycoprops

def compute_roi_boundaries(rfp_image, x, y, scales, octaves):
    """
    Compute ROI boundaries for arrays of x, y, scales, and octaves.
    
    The ROI is a square centered at (x,y) with half-size given by
      radius = scales * (2 ** (octaves + 1)).
    """
    radii = scales * (2 ** (octaves + 1))
    xmin = np.clip(np.floor(x - radii).astype(int), 0, rfp_image.shape[0])
    xmax = np.clip(np.ceil(x + radii).astype(int), 0, rfp_image.shape[0])
    ymin = np.clip(np.floor(y - radii).astype(int), 0, rfp_image.shape[1])
    ymax = np.clip(np.ceil(y + radii).astype(int), 0, rfp_image.shape[1])
    return xmin, xmax, ymin, ymax

def compute_mean_intensities(rfp_image, df):
    """
    Compute mean intensity for each ROI in the DataFrame using an integral image.
    
    Parameters:
      rfp_image : 2D numpy array for the RFP channel.
      df        : DataFrame containing columns 'x', 'y', 'scales', and 'octaves'.
      
    Returns:
      A list of mean intensity values (one per ROI).
    """
    # Extract ROI parameters as arrays.
    x = df['x'].values
    y = df['y'].values
    scales = df['scales'].values
    octaves = df['octaves'].values
    
    xmin, xmax, ymin, ymax = compute_roi_boundaries(rfp_image, x, y, scales, octaves)
    
    # Compute the integral image once.
    ii = integral_image(rfp_image)
    
    mean_intensity = []
    for i in range(len(x)):
        x_min = xmin[i]
        x_max = xmax[i]
        y_min = ymin[i]
        y_max = ymax[i]
        area = (x_max - x_min) * (y_max - y_min)
        if area <= 0:
            mean_intensity.append(0)
        else:
            # Compute the sum using four look-ups.
            A = ii[x_max-1, y_max-1] if (x_max-1 >= 0 and y_max-1 >= 0) else 0
            B = ii[x_min-1, y_max-1] if x_min-1 >= 0 else 0
            C = ii[x_max-1, y_min-1] if y_min-1 >= 0 else 0
            D = ii[x_min-1, y_min-1] if (x_min-1 >= 0 and y_min-1 >= 0) else 0
            sum_intensity = A - B - C + D
            mean_intensity.append(sum_intensity / area)
    return mean_intensity


def compute_rfp_pos_frac_for_roi(rfp_image, row, threshold=3.5):
    """
    Compute the fraction of RFP positive pixels for an ROI in the DataFrame.
    """
    radius = row['scales'] * (2 ** (row['octaves'] + 1))
    x_min = np.clip(np.floor(row['x'] - radius).astype(int), 0, rfp_image.shape[0])
    x_max = np.clip(np.ceil(row['x'] + radius).astype(int), 0, rfp_image.shape[0])
    y_min = np.clip(np.floor(row['y'] - radius).astype(int), 0, rfp_image.shape[1])
    y_max = np.clip(np.ceil(row['y'] + radius).astype(int), 0, rfp_image.shape[1])
    roi = rfp_image[x_min:x_max, y_min:y_max]
    return np.sum(roi > threshold) / (x_max - x_min) / (y_max - y_min)


def compute_rfp_pos_fractions(rfp_image, df, threshold=3.5):
    """
    Compute RFP positive pixel fractions for all ROIs in the DataFrame.
    
    Uses a DataFrame.apply call to process each ROI.
    """
    return df.apply(lambda row: compute_rfp_pos_frac_for_roi(rfp_image, row, threshold), axis=1)

def compute_glcm_for_roi(image, row):
    """
    Compute the GLCM for a single ROI defined in the row.
    """
    radius = row['scales'] * (2 ** (row['octaves'] + 1))
    x_min = int(np.clip(np.floor(row['x'] - radius), 0, image.shape[0]))
    x_max = int(np.clip(np.ceil(row['x'] + radius), 0, image.shape[0]))
    y_min = int(np.clip(np.floor(row['y'] - radius), 0, image.shape[1]))
    y_max = int(np.clip(np.ceil(row['y'] + radius), 0, image.shape[1]))
    roi = image[x_min:x_max, y_min:y_max]

    texture_mat = graycomatrix(roi,
                                distances=[1, 2],
                                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=256,
                                symmetric=True,
                                normed=True)
    glcm_homogeneity = graycoprops(texture_mat, 'homogeneity')[0, 0]
    glcm_energy = graycoprops(texture_mat, 'energy')[0, 0]

    return glcm_homogeneity, glcm_energy

def compute_glcm(image, df):
    """
    Compute the GLCM for all ROIs in the DataFrame.
    
    Uses a DataFrame.apply call to process each ROI.
    """
    results = df.apply(lambda row: compute_glcm_for_roi(image, row), axis=1)
    glcm_homogeneities, glcm_energies = zip(*results)
    return list(glcm_homogeneities), list(glcm_energies)

def morans_i(image, connectivity=8):
    """
    Compute Moran's I spatial autocorrelation for a 2D image.

    Parameters:
      image : 2D numpy array
          The input image (e.g., intensity values).
      connectivity : int, optional (default=8)
          Determines the neighbor connectivity:
            - If 4, only consider up, down, left, right neighbors.
            - If 8, include diagonals as well.

    Returns:
      I : float
          Moran's I statistic.
    
    Notes:
      - The function uses a convolution-based approach to compute the spatial lag.
      - Border pixels are treated with a zero-fill (thus having fewer neighbors).
    """
    # Number of pixels
    n = image.size

    # Compute deviations from the mean
    z = image - np.mean(image)
    
    # Define weight kernel based on connectivity.
    if connectivity == 4:
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])
    else:  # default to 8-neighbor connectivity
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]])
    
    # Compute the spatially lagged values via convolution.
    # This computes, for each pixel, the weighted sum of its neighbors' deviations.
    weighted_sum = convolve2d(z, kernel, mode='same', boundary='fill', fillvalue=0)
    
    # Numerator: sum over pixels of z_i multiplied by the weighted sum of neighbors.
    numerator = np.sum(z * weighted_sum)
    
    # Compute S0: the sum of the weights for each pixel.
    # Since border pixels have fewer neighbors, convolve an array of ones.
    ones = np.ones_like(image)
    weights_per_pixel = convolve2d(ones, kernel, mode='same', boundary='fill', fillvalue=0)
    S0 = np.sum(weights_per_pixel)
    
    # Denominator: sum of squared deviations.
    denominator = np.sum(z**2)
    
    # Compute Moran's I using the standard formula.
    I = (n / S0) * (numerator / denominator)
    return I

def compute_morans_I_for_roi(image, row):
    """
    Compute Moran's I for a single ROI defined in the row.
    """
    radius = row['scales'] * (2 ** (row['octaves'] + 1))
    x_min = int(np.clip(np.floor(row['x'] - radius), 0, image.shape[0]))
    x_max = int(np.clip(np.ceil(row['x'] + radius), 0, image.shape[0]))
    y_min = int(np.clip(np.floor(row['y'] - radius), 0, image.shape[1]))
    y_max = int(np.clip(np.ceil(row['y'] + radius), 0, image.shape[1]))
    roi = image[x_min:x_max, y_min:y_max]
    return morans_i(roi)

def compute_morans_Is(image, df):
    """
    Compute the Moran's I for all ROIs in the DataFrame.
    
    Uses a DataFrame.apply call to process each ROI.
    """
    return df.apply(lambda row: compute_morans_I_for_roi(image, row), axis=1)
