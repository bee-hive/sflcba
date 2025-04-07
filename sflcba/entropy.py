# sflcba/sift.py
import numpy as np

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
