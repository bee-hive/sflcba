# sflcba/sift.py
import numpy as np
import cv2 as cv
import tiffile as tiff
from skimage.feature import SIFT


def compute_sift_embedding(image_path, downsample_pct=0.1):
    """
    Given the path to a phase image file, compute the SIFT keypoints and descriptors.
    This function loads the image, runs the SIFT detector, and downsamples the results.
    Returns a dictionary with the embedding and associated metadata.
    """
    # Load image and normalize intensity
    image = tiff.imread(image_path)
    phase_frame = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    
    # Detect SIFT keypoints and compute descriptors
    descriptor_extractor = SIFT()
    descriptor_extractor.detect_and_extract(phase_frame)
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors
    scales = descriptor_extractor.scales
    octaves = descriptor_extractor.octaves
    sigmas = descriptor_extractor.sigmas
    orientations = descriptor_extractor.orientations
    
    # Downsample descriptors to reduce memory footprint
    n_keypoints = len(keypoints)
    idx = np.random.choice(n_keypoints, int(n_keypoints * downsample_pct), replace=False)
    
    # Package results in a dictionary (you can also construct an AnnData object here)
    result = {
        "x": keypoints[idx, 0],
        "y": keypoints[idx, 1],
        "descriptors": descriptors[idx],
        "scales": scales[idx],
        "octaves": octaves[idx],
        "sigmas": sigmas[idx],
        "orientations": orientations[idx],
        "n_keypoints_original": n_keypoints
    }
    return result
