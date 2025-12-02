---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import tiffile as tiff
import glob
import re
import random
import anndata as ad
import pandas as pd
import scipy
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('default')
import os
from scipy.ndimage import gaussian_filter
from sflcba.entropy import entropy

random.seed(0)
np.random.seed(0)
```

```python
SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.use14corefonts'] = True
```

### Simulate RFP masks with known spatial entropy coefficients

```python
# find the size of the x and y dimensions of the images
# load the anndata file with the SIFT descriptors
adata = ad.read_h5ad('analysis/adata_20250225_processed_20250310.h5ad')
x_min, x_max = int(adata.obs['x'].min()), int(adata.obs['x'].max())
y_min, y_max = int(adata.obs['y'].min()), int(adata.obs['y'].max())
print(f"Image dimensions: x: {x_min} to {x_max}, y: {y_min} to {y_max}")
```

```python

def draw_cell(img, x, y, radius):
    """Draw a filled circle (cell) onto img at (x,y)."""
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = (xx - x)**2 + (yy - y)**2 <= radius**2
    img[mask] = 1
    return img

def simulate_cells(alpha, 
                   size=1000, 
                   cell_radius=4, 
                   base_clusters=200, 
                   base_cells_per_cluster=20):
    """
    Generate a 1000x1000 binary image containing cell-like objects
    whose spatial aggregation level is controlled by alpha.

    alpha = 1.0 → high entropy → cells scattered independently
    alpha = 0.1 → low entropy → strong clustering
    """

    img = np.zeros((size, size), dtype=np.uint8)

    # Number of cluster centers:
    # high alpha → many clusters → cells look dispersed
    n_clusters = int(base_clusters * alpha + 1)

    # Cells per cluster:
    # low alpha → many cells per cluster → aggregates
    cells_per_cluster = int(base_cells_per_cluster * (1/alpha))

    # Spatial dispersion around cluster center:
    # high alpha → wider spread → more randomness
    cluster_spread = 50 * alpha + 1   # pixels

    # Sample cluster centers uniformly
    cluster_centers = np.random.randint(0, size, size=(n_clusters, 2))

    for cx, cy in cluster_centers:
        for _ in range(cells_per_cluster):

            # position relative to cluster center
            x = int(cx + np.random.randn() * cluster_spread)
            y = int(cy + np.random.randn() * cluster_spread)

            # ensure in bounds
            if 0 <= x < size and 0 <= y < size:
                draw_cell(img, x, y, cell_radius)

    return img

alpha = 0.3   # fairly low entropy → pronounced aggregates
images = [simulate_cells(alpha) for _ in range(10)]

```

```python
fig, ax = plt.subplots(2, 5, figsize=(10, 6), tight_layout=True)
ax = ax.ravel()
for i in range(10):
    ax[i].axis('off')
    ax[i].set_title(f'Simulated Image {i+1} (alpha={alpha})')
    ax[i].imshow(images[i], cmap='Reds', alpha = 1.0)
plt.show()
```

```python
alpha = 0.75   # example: high entropy
images = [simulate_cells(alpha) for _ in range(10)]
fig, ax = plt.subplots(2, 5, figsize=(10, 6), tight_layout=True)
ax = ax.ravel()
for i in range(10):
    ax[i].axis('off')
    ax[i].set_title(f'Simulated Image {i+1} (alpha={alpha})')
    ax[i].imshow(images[i], cmap='Reds', alpha = 1.0)
plt.show()
```

```python
# compute the entropy of one of the simulated images
sim_image = images[0]
sim_entropy = np.sum(entropy(sim_image))
print(f"Simulated image entropy: {sim_entropy}")
```

### Simulate 50 different RFP masks with spatial entropy coefficents ranging from 0.3 to 0.9, computing the entropy of each image

Add the image ID, true alpham and computed entropy to a dataframe. Store each raw image in a list where the element is the image ID.

```python
images_list = []
entropy_list = []
alpha_list = []
for alpha in np.linspace(0.2, 0.8, 50):
    sim_img = simulate_cells(alpha)
    sim_ent = np.sum(entropy(sim_img))
    images_list.append(sim_img)
    entropy_list.append(sim_ent)
    alpha_list.append(alpha)

df = pd.DataFrame({
    'image_id': range(len(images_list)),
    'alpha': alpha_list,
    'entropy': entropy_list
})
df.head()
```

```python
# plot true alpha vs computed entropy
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='alpha', y='entropy')
plt.title('Simulated Image Entropy vs True Alpha')
plt.xlabel('True Alpha (Spatial Entropy Coefficient)')
plt.ylabel('Computed Entropy')
plt.show()
```

### Benchmark against different ways to compute spatial entropy


#### 1. Quadrat-count entropy (generalization of your block-entropy)

This is equivalent to Shannon entropy over quadrat cell-counts.

```python
def quadrat_entropy(binary_image, block_size=(100,100)):
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
```

#### 2. Nearest-Neighbor Distance (NND) Entropy

Measures disorder in how cells are spaced.
Cells are treated as points at the centers of their pixels.

```python
from scipy.spatial import cKDTree

def nearest_neighbor_entropy(binary_image, bins=50):
    coords = np.column_stack(np.where(binary_image == 1))
    if len(coords) < 2:
        return 0.0

    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    nnd = dists[:, 1]

    # Histogram of counts
    hist, _ = np.histogram(nnd, bins=bins)
    
    # Convert to probabilities
    p = hist / hist.sum()
    p = p[p > 0]

    return -np.sum(p * np.log(p))


```

#### 3. Patch-Size Entropy

Entropy over the sizes of connected components of 1-pixels → cluster-size diversity.

```python
from scipy.ndimage import label

def patch_size_entropy(binary_image):
    labeled, n = label(binary_image)
    if n == 0:
        return 0.0

    sizes = np.bincount(labeled.ravel())[1:]  # skip background
    p = sizes / sizes.sum()
    return -np.sum(p * np.log(p))

```

#### 4. GLCM (Co-occurrence) Entropy

Captures pixel-level 2nd-order texture.
Useful for fine-grained heterogeneity.

Binary images → two gray levels {0,1}.
We compute a simple horizontal GLCM.

```python
from sklearn.feature_extraction import image

def glcm_entropy(binary_image, dx=1, dy=0):
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

```

#### 5. Multiscale Block Entropy

Entropy is computed at multiple resolutions and summed/averaged.

Useful for tissues that have aggregates across several spatial scales.

```python
def multiscale_entropy(binary_image, block_sizes=[20, 50, 100, 200]):
    entropies = []
    for b in block_sizes:
        entropies.append(quadrat_entropy(binary_image, (b, b)))
    return np.array(entropies)

```

#### 6. Ripley’s L-function Deviation Score

Not an entropy, but the gold standard for spatial clustering.
We convert it into a single “entropy-like” statistic by integrating deviations from CSR.

```python
def ripley_L_deviation(binary_image, radii=np.linspace(5,150,30)):
    coords = np.column_stack(np.where(binary_image == 1))
    n = len(coords)
    if n < 2:
        return 0.0

    area = binary_image.size
    density = n / area

    tree = cKDTree(coords)
    deviations = []

    for r in radii:
        counts = tree.query_ball_point(coords, r)
        k_r = np.mean([len(c)-1 for c in counts]) / density
        L_r = np.sqrt(k_r / np.pi)
        deviations.append(np.abs(L_r - r))

    return np.sum(deviations)

```

```python
# compute the value of each entropy metric for all simulated images
results = []
for idx, img in enumerate(images_list):
    # find the true alpha and compute entropy for this image
    temp_alpha = alpha_list[idx]
    temp_entropy = entropy_list[idx]
    # compute all entropy metrics
    q_entropy = quadrat_entropy(img, block_size=(100,100))
    nnd_entropy = nearest_neighbor_entropy(img)
    psize_entropy = patch_size_entropy(img)
    glcm_ent = glcm_entropy(img)
    mscale_ent = multiscale_entropy(img).mean()  # average over scales
    # ripley = ripley_L_deviation(img)

    results.append({
        'image_id': idx,
        'alpha': temp_alpha,
        'computed_entropy': temp_entropy,
        'quadrat_entropy': q_entropy,
        'nnd_entropy': nnd_entropy,
        'patch_size_entropy': psize_entropy,
        'glcm_entropy': glcm_ent,
        'multiscale_entropy': mscale_ent,
        # 'ripley_L_deviation': ripley
    })
results_df = pd.DataFrame(results)
results_df.head()
```

```python
# plot true alpha vs computed entropy for each metric
metrics = ['quadrat_entropy', 'patch_size_entropy', 'glcm_entropy', 'multiscale_entropy']
fig, ax = plt.subplots(1, 4, figsize=(12, 3), tight_layout=True)
ax = ax.ravel()
for i, metric in enumerate(metrics):
    sns.scatterplot(data=results_df, x='alpha', y=metric, ax=ax[i])
    # compute pearson correlation
    corr, pval = scipy.stats.spearmanr(results_df['alpha'], results_df[metric])
    # annotate correlation
    ax[i].annotate(f'Spearman r={corr:.2f}\np={pval:.2e}', 
                   xy=(0.95, 0.1), xycoords='axes fraction',
                   fontsize=8, ha='right', va='center')
    ax[i].set_title(f'{metric.replace("_", " ").title()}')
    ax[i].set_xlabel('True Alpha (Spatial Entropy Coefficient)\n<-- structured | random -->')
    ax[i].set_ylabel(metric.replace("_", " ").title())
plt.show()
```

### Create an alternative simulator that has block-level spatial heterogeneity

```python

def simulate_block_entropy_image(
        alpha,
        image_size=1000,
        block_size=100,
        total_cells=5000,
        cell_radius=4):
    """
    Simulate a binary 2D image where macro-scale heterogeneity is explicitly
    controlled by alpha via a Dirichlet distribution over block densities.
    """

    H, W = image_size, image_size
    bh, bw = block_size, block_size
    n_blocks_h = H // bh
    n_blocks_w = W // bw
    B = n_blocks_h * n_blocks_w

    # 1. Dirichlet over block weights
    # alpha controls heterogeneity: small alpha → spiky → low entropy
    block_probs = np.random.dirichlet(alpha * np.ones(B))

    # 2. Allocate cell counts to blocks
    block_counts = np.random.multinomial(total_cells, block_probs)

    # 3. Prepare output image
    img = np.zeros((H, W), dtype=np.uint8)

    # 4. Function to draw a filled circle
    def draw_cell_local(img, x, y, r):
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - x)**2 + (yy - y)**2 <= r**2
        img[mask] = 1

    # 5. Populate blocks with cells according to allocated counts
    block_index = 0
    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            count = block_counts[block_index]
            block_index += 1

            # block coordinate ranges
            x0, x1 = j * bw, (j+1) * bw
            y0, y1 = i * bh, (i+1) * bh

            # generate count random locations inside the block
            xs = np.random.randint(x0, x1, size=count)
            ys = np.random.randint(y0, y1, size=count)

            for x, y in zip(xs, ys):
                draw_cell_local(img, y, x, cell_radius)

    return img

```

```python
alpha = 0.2   # low entropy → pronounced block-level heterogeneity
block_images = [simulate_block_entropy_image(alpha=alpha) for _ in range(10)]
fig, ax = plt.subplots(2, 5, figsize=(10, 6), tight_layout=True)
ax = ax.ravel()
for i in range(10):
    ax[i].axis('off')
    ax[i].set_title(f'Simulated Image {i+1} (alpha={alpha})')
    ax[i].imshow(block_images[i], cmap='Reds', alpha = 1.0)
plt.show()
```

```python
alpha = 3   # low entropy → pronounced block-level heterogeneity
block_images = [simulate_block_entropy_image(alpha=alpha) for _ in range(2)]
fig, ax = plt.subplots(2, 5, figsize=(10, 6), tight_layout=True)
ax = ax.ravel()
for i in range(2):
    ax[i].axis('off')
    ax[i].set_title(f'Simulated Image {i+1} (alpha={alpha})')
    ax[i].imshow(block_images[i], cmap='Reds', alpha = 1.0)
plt.show()
```

```python
# simulate block entropy images with varying alpha and compute all entropy metrics
results_block = []

for i, alpha in enumerate(np.linspace(0.2, 3, 50)):
    sim_img_block = simulate_block_entropy_image(alpha)
    sim_ent_block = np.sum(entropy(sim_img_block))

    # compute the other entropy metrics as well
    q_entropy = quadrat_entropy(sim_img_block, block_size=(100,100))
    nnd_entropy = nearest_neighbor_entropy(sim_img_block)
    psize_entropy = patch_size_entropy(sim_img_block)
    glcm_ent = glcm_entropy(sim_img_block)
    mscale_ent = multiscale_entropy(sim_img_block).mean()  # average over scales

    results_block.append({
        'image_id': i,
        'alpha': alpha,
        'computed_entropy': sim_ent_block,
        'quadrat_entropy': q_entropy,
        'nnd_entropy': nnd_entropy,
        'patch_size_entropy': psize_entropy,
        'glcm_entropy': glcm_ent,
        'multiscale_entropy': mscale_ent,
    })
results_block_df = pd.DataFrame(results_block)
results_block_df.head()
```

```python
# plot true alpha vs computed entropy for each metric
metrics = ['quadrat_entropy', 'patch_size_entropy', 'glcm_entropy', 'multiscale_entropy']
fig, ax = plt.subplots(1, 4, figsize=(12, 3), tight_layout=True)
ax = ax.ravel()
for i, metric in enumerate(metrics):
    sns.scatterplot(data=results_block_df, x='alpha', y=metric, ax=ax[i])
    # compute pearson correlation
    corr, pval = scipy.stats.spearmanr(results_block_df['alpha'], results_block_df[metric])
    # annotate correlation
    ax[i].annotate(f'Spearman r={corr:.2f}\np={pval:.2e}', 
                   xy=(0.95, 0.1), xycoords='axes fraction',
                   fontsize=8, ha='right', va='center')
    ax[i].set_title(f'{metric.replace("_", " ").title()}')
    ax[i].set_xlabel('True Alpha (Spatial Entropy Coefficient)\n<-- structured | random -->')
    ax[i].set_ylabel(metric.replace("_", " ").title())
plt.show()
```

```python
fig, ax = plt.subplots(3, 4, figsize=(10, 7.5), tight_layout=True)
ax = ax.ravel()

# plot a low entropy example and a high entropy example from the point cloud simulator
for i in range(2):
    if i == 0:
        alpha = 0.2
    else:
        alpha = 0.8
    sim_img = simulate_cells(alpha)
    ax[i].axis('off')
    ax[i].set_title(f'Point cloud simulator (alpha={alpha})')
    ax[i].imshow(sim_img, cmap='Reds', alpha = 1.0)
# plot a low entropy example and a high entropy example from the block simulator
for i in range(2):
    if i == 0:
        zeta = 0.2
    else:
        zeta = 3.0
    sim_img = simulate_block_entropy_image(zeta)
    ax[i+2].axis('off')
    ax[i+2].set_title(f'Block simulator (zeta={zeta})')
    ax[i+2].imshow(sim_img, cmap='Reds', alpha = 1.0)

# plot true alpha vs computed entropy for each metric of the point cloud simulator results
metrics = ['quadrat_entropy', 'patch_size_entropy', 'glcm_entropy', 'multiscale_entropy']
for i, metric in enumerate(metrics):
    offset = 4
    sns.scatterplot(data=results_df, x='alpha', y=metric, ax=ax[i+offset])
    # compute pearson correlation
    corr, pval = scipy.stats.spearmanr(results_df['alpha'], results_df[metric])
    # annotate correlation
    ax[i+offset].annotate(f'Spearman r={corr:.2f}\np={pval:.2e}', 
                   xy=(0.95, 0.1), xycoords='axes fraction',
                   fontsize=8, ha='right', va='center')
    ax[i+offset].set_title(f'Point cloud simulator')
    ax[i+offset].set_xlabel('True alpha\n<-- structured | random -->')
    ax[i+offset].set_ylabel(metric.replace("_", " ").title())

# plot the true zeta vs computed entropy for each metric of the block simulator results
for i, metric in enumerate(metrics):
    offset = 8
    sns.scatterplot(data=results_block_df, x='alpha', y=metric, ax=ax[i+offset])
    # compute pearson correlation
    corr, pval = scipy.stats.spearmanr(results_block_df['alpha'], results_block_df[metric])
    # annotate correlation
    ax[i+offset].annotate(f'Spearman r={corr:.2f}\np={pval:.2e}', 
                   xy=(0.95, 0.1), xycoords='axes fraction',
                   fontsize=8, ha='right', va='center')
    ax[i+offset].set_title(f'Block simulator')
    ax[i+offset].set_xlabel('True zeta\n<-- structured | random -->')
    ax[i+offset].set_ylabel(metric.replace("_", " ").title())


fig.savefig('figures/fig1/entropy_simulations.pdf', dpi=300, bbox_inches='tight')

plt.show()
```

```python

```
