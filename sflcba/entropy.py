# sflcba/entropy.py
import numpy as np
from scipy.ndimage import label

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

def entropy(binary_image, block_size=(100,100)):
    """
    Compute entropy of a binary image over blocks.
    """
    cancer_cells = np.sum(binary_image)
    blocks = blockshaped(binary_image, block_size[0], block_size[1])
    entropies = []
    for block in blocks:
        num = np.sum(block)
        if num != 0:
            p = num / cancer_cells
            entropies.append(-p * np.log(p))
        else:
            entropies.append(0)
    return entropies

def quadrat_entropy(binary_image, block_size=(100,100)):
    """
    Compute quadrat entropy of a binary image. This formula is equivalent to Shannon entropy over quadrat cell counts.
    """
    H, W = binary_image.shape
    bh, bw = block_size
    blocks = []

    # Extract blocks
    for i in range(0, H, bh):
        for j in range(0, W, bw):
            block = binary_image[i:i+bh, j:j+bw]
            blocks.append(np.sum(block))

    blocks = np.array(blocks)
    total = blocks.sum()
    
    if total == 0:
        return 0.0

    p = blocks / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def patch_size_entropy(binary_image):
    """
    Compute patch size entropy of a binary image. This formula computes entropy over the size of 1-pixel connected components.
    """
    labeled, n = label(binary_image)
    if n == 0:
        return 0.0

    sizes = np.bincount(labeled.ravel())[1:]  # skip background
    p = sizes / sizes.sum()
    return -np.sum(p * np.log(p))


def glcm_entropy(binary_image, dx=1, dy=0):
    """
    Compute grey level co-occurrence matrix (GLCM) entropy of a binary image for a given offset (dx, dy).
    """
    H, W = binary_image.shape
    cooc = np.zeros((2,2), float)

    for i in range(H):
        for j in range(W):
            i2, j2 = i + dy, j + dx
            if 0 <= i2 < H and 0 <= j2 < W:
                cooc[binary_image[i,j], binary_image[i2,j2]] += 1

    if cooc.sum() == 0:
        return 0.0

    p = cooc / cooc.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def multiscale_quadrat_entropy(binary_image, block_sizes=[20, 50, 100, 200]):
    """
    Compute multiscale quadrat entropy of a binary image. Return the entropies for multiple block sizes.
    It is common to take the mean across block sizes as a single summary metric.
    """
    entropies = []
    for b in block_sizes:
        entropies.append(quadrat_entropy(binary_image, (b, b)))
    return np.array(entropies)
