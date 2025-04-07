# sflcba/sift.py
import numpy as np
import cv2 as cv
import glob
import re
import random
from skimage.feature import SIFT as sift

def blockshaped(arr, nrows, ncols):
    """
    Break a 2D array into blocks of shape (nrows, ncols).
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def entropy(binary_image):
    """
    Compute entropy of a binary image over blocks.
    """
    cancer_cells = np.sum(binary_image)
    blocks = blockshaped(binary_image, 100, 100)
    entropies = []
    for block in blocks:
        num = np.sum(block)
        if num != 0:
            p = num / cancer_cells
            entropies.append(-p * np.log(p))
        else:
            entropies.append(0)
    return entropies

def sorted_nicely(l):
    """Sort the given iterable in human order."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

def compute_sift_embedding(image_path, downsample_pct=0.1):
    """
    Given the path to a phase image file, compute the SIFT keypoints and descriptors.
    This function loads the image, runs the SIFT detector, and downsamples the results.
    Returns a dictionary with the embedding and associated metadata.
    """
    # Load image and normalize intensity
    import tiffile as tiff
    image = tiff.imread(image_path)
    phase_frame = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    
    # Detect SIFT keypoints and compute descriptors
    descriptor_extractor = sift()
    descriptor_extractor.detect_and_extract(phase_frame)
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors
    scales = descriptor_extractor.scales
    octaves = descriptor_extractor.octaves
    orientations = descriptor_extractor.orientations
    
    # Downsample descriptors to reduce memory footprint
    n_keypoints = len(keypoints)
    idx = np.random.choice(n_keypoints, int(n_keypoints * downsample_pct), replace=False)
    
    # Package results in a dictionary (you can also construct an AnnData object here)
    result = {
        "keypoints": keypoints[idx],
        "descriptors": descriptors[idx],
        "scales": scales[idx],
        "octaves": octaves[idx],
        "orientations": orientations[idx],
        "n_original": n_keypoints
    }
    return result
