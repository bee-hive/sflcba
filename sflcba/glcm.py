# sflcba/glcm.py
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import integral_image
import cv2 as cv

def compute_glcm_for_roi(roi, distances=[1,2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Compute the grey-level co-occurrence matrix for a given ROI and return selected properties.
    """
    glcm = graycomatrix(roi, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    return homogeneity, energy

def compute_mean_intensity(rfp_image, x, y, scales, octaves):
    """
    Compute the mean intensity of the RFP channel in an ROI.
    """
    # Define ROI boundaries based on scale and octave
    radius = scales * (2 ** (octaves + 1))
    x_min = int(max(np.floor(x - radius), 0))
    x_max = int(min(np.ceil(x + radius), rfp_image.shape[0]))
    y_min = int(max(np.floor(y - radius), 0))
    y_max = int(min(np.ceil(y + radius), rfp_image.shape[1]))
    roi = rfp_image[x_min:x_max, y_min:y_max]
    area = (x_max - x_min) * (y_max - y_min)
    return np.sum(roi) / area if area > 0 else 0
