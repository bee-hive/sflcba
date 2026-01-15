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
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns
import tiffile as tiff
import random
import anndata as ad
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score
import scipy.stats as stats
from sflcba.stats import cluster_enrichment
from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering

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

```python
# load the anndata file with the SIFT descriptors
# adata = ad.read_h5ad('analysis/adata_processed.h5ad')
adata = ad.read_h5ad('analysis/adata_20250225_processed_20250310.h5ad')

# the number of rows when first loading the data represents the number of SIFT descriptors used for clustering
num_rows_cluster = adata.obs.shape[0]

adata
```

```python
# image spatialresolution is 4.975 um per pixel
um_per_pixel = 4.975

# convert the x and y coordinates from pixels to um
adata.obs['x_um'] = adata.obs['x'] * um_per_pixel
adata.obs['y_um'] = adata.obs['y'] * um_per_pixel

# convert the p_area from pixels^2 to μm$^2$
adata.obs['p_areas'] = adata.obs['p_areas'] * (um_per_pixel**2)
```

```python
# image time resolution is 2 hours per frame
hours_per_frame = 2

# convert the time from frames to hours
adata.obs['time'] = adata.obs['time'] * hours_per_frame
```

```python
# move PC1 and PC2 from adata.obsm['X_pca'] to adata.obs['PC1'] and adata.obs['PC2']
adata.obs['PC1'] = adata.obsm['X_pca'][:, 0]
adata.obs['PC2'] = adata.obsm['X_pca'][:, 1]
adata.obs['PC3'] = adata.obsm['X_pca'][:, 2]
adata.obs['PC4'] = adata.obsm['X_pca'][:, 3]
```

### Perform K-means clustering with k=7 but different random seeds

```python

def cluster_sift_embedding(adata, k, random_state=0, label_key=None):
    """
    Given an AnnData object with SIFT embeddings in .X and metadata in .obs,
    perform k-means clustering for each k in k_values.
    The cluster labels and the within-cluster sum of squares (wccs) for each k
    are added to the AnnData object.
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state).fit(adata.X)
    if label_key is None:
        label_key = f'kmeans_{k}'
    adata.obs[label_key] = kmeans.labels_
    adata.obs[label_key] = adata.obs[label_key].astype('category')
    if not hasattr(adata, 'uns'):
        adata.uns = {}
    adata.uns[label_key] = {'wccs': kmeans.inertia_}
    return adata

for seed in range(1, 4):
    print(f'Clustering SIFT embeddings with k=7, seed={seed}...')
    adata = cluster_sift_embedding(adata, k=7, random_state=seed, label_key=f'kmeans_7_seed{seed}')



```

```python
adata.obs.columns
```

```python
# make a copy of the entire adata object before subsetting
adata_full = adata.copy()

# subset the entire adata object to just 50k randomly sampled rows
# this should help with runtime issues and overpowered statistical testing
num_rows = 5000
# num_rows = adata.obs.shape[0]
adata = adata[np.random.choice(adata.shape[0], num_rows, replace=False), :]
adata.obs.reset_index(drop=True, inplace=True)
adata
```

```python
# plot PC1 vs PC2 colored by k-means cluster for each different random seed
fig, ax = plt.subplots(1, 3, figsize=(9, 3), tight_layout=True)
ax = ax.ravel()

sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue=adata.obs[f'kmeans_7'], palette='Dark2', alpha=0.7, s=3, ax=ax[0], rasterized=True)
for i in range(1,3):
    sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue=adata.obs[f'kmeans_7_seed{i}'], palette='Dark2', alpha=0.7, s=3, ax=ax[i], rasterized=True)

for i in range(3):
    ax[i].set_xlabel('PC1')
    ax[i].set_ylabel('PC2')
    sns.despine(ax=ax[i])
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    # compare the Adjusted Rand Index between the clusters
    if i == 0:
        ax[i].set_title(f'k-means k=7, seed=0')
    else:
        ari = adjusted_rand_score(adata.obs[f'kmeans_7'], adata.obs[f'kmeans_7_seed{i}'])
        ax[i].set_title(f'k-means k=7, seed={i}\nARI={ari:.2f}')
    ax[i].legend(title='Cluster', markerscale=3)

fig.savefig('figures/fig2/kmeans_7_seeds.pdf', bbox_inches='tight', dpi=300)

plt.show()
```

```python
cm.get_cmap('Dark2', 8)
```

```python
# plot a clustered heatmap of N=100 rows of the SIFT descriptor matrix
# select 100 random rows from adata.X
num_heatmap_rows = 200
# pick a set of random indices without replacement so I can extract heatmap and cluster ID labels
heatmap_rows = np.random.choice(adata.shape[0], num_heatmap_rows, replace=False)
sift_subset = adata.X[heatmap_rows, :]
kmeans_labels = adata.obs[f'kmeans_7'].values[heatmap_rows]
# map the kmeans labels to colors using the Dark2 colormap
cmap = cm.get_cmap('Dark2', 8)
kmeans_colors = [cmap(label) for label in kmeans_labels]

# plot a clustered heatmap of the sift_subset
sns.clustermap(sift_subset, row_colors=kmeans_colors, method='average', metric='euclidean', cmap='viridis', figsize=(6, 6), rasterized=True, )
# plt.savefig('figures/fig2/sift_descriptor_heatmap.pdf', bbox_inches='tight', dpi=300)
plt.show()
```

### Perform hdbscan clustering of adata_full and compare the results to k-means clustering

```python
def cluster_sift_embedding_hdbscan(adata, label_key=None, min_cluster_size=10, metric='euclidean'):
    """
    Given an AnnData object with SIFT embeddings in .X and metadata in .obs,
    perform hdbscan clustering according to the specified parameters.
    The cluster labels and the within-cluster sum of squares (wccs) for each k
    are added to the AnnData object.
    """
    clust = HDBSCAN(min_cluster_size=min_cluster_size, metric=metric, max_cluster_size=1000).fit(adata.X)
    if label_key is None:
        label_key = f'hdbscan_min{min_cluster_size}_{metric}'
    adata.obs[label_key] = clust.labels_
    adata.obs[label_key] = adata.obs[label_key].astype('category')
    return adata

for metric in ['euclidean', 'chebyshev', 'manhattan']:
    print(f'Clustering SIFT embeddings with HDBSCAN, metric={metric}, min_cluster_size=10...')
    adata = cluster_sift_embedding_hdbscan(adata, label_key=f'hdbscan_min10_{metric}', min_cluster_size=10, metric=metric)
```

```python
# plot PC1 vs PC2 colored by k-means cluster for each different random seed
fig, ax = plt.subplots(1, 3, figsize=(9, 3), tight_layout=True)
ax = ax.ravel()

sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue=adata.obs[f'hdbscan_min10_euclidean'], palette='Dark2', alpha=0.7, s=3, ax=ax[0], rasterized=True)
ax[0].set_title('HDBSCAN metric=euclidean')
sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue=adata.obs[f'hdbscan_min10_chebyshev'], palette='Dark2', alpha=0.7, s=3, ax=ax[1], rasterized=True)
# compute the Adjusted Rand Index between euclidean and chebyshev
ari = adjusted_rand_score(adata.obs[f'hdbscan_min10_euclidean'], adata.obs[f'hdbscan_min10_chebyshev'])
ax[1].set_title('HDBSCAN metric=chebyshev\nvs euclidean ARI={:.2f}'.format(ari))
sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue=adata.obs[f'hdbscan_min10_manhattan'], palette='Dark2', alpha=0.7, s=3, ax=ax[2], rasterized=True)
# compute the Adjusted Rand Index between euclidean and manhattan
ari = adjusted_rand_score(adata.obs[f'hdbscan_min10_euclidean'], adata.obs[f'hdbscan_min10_manhattan'])
ax[2].set_title('HDBSCAN metric=manhattan\nvs euclidean ARI={:.2f}'.format(ari))

for i in range(3):
    ax[i].set_xlabel('PC1')
    ax[i].set_ylabel('PC2')
    sns.despine(ax=ax[i])
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].legend(title='Cluster', markerscale=3)

fig.savefig('figures/fig2/hdbscan_comparison.pdf', bbox_inches='tight', dpi=300)

plt.show()
```

```python


def cluster_sift_embedding_Agglomerative(adata, label_key=None, k=7, metric='euclidean'):
    """
    Given an AnnData object with SIFT embeddings in .X and metadata in .obs,
    perform hdbscan clustering according to the specified parameters.
    The cluster labels and the within-cluster sum of squares (wccs) for each k
    are added to the AnnData object.
    """
    clust = AgglomerativeClustering(n_clusters=k, metric='euclidean').fit(adata.X)
    if label_key is None:
        label_key = f'Agglomerative_k{k}'
    adata.obs[label_key] = clust.labels_
    adata.obs[label_key] = adata.obs[label_key].astype('category')
    return adata

adata = cluster_sift_embedding_Agglomerative(adata, label_key=f'Agglomerative_euclidean_k7', k=7)


# compare the k-means to agglomerative clustering and hdbscan clustering w/ euclidean metric
fig, ax = plt.subplots(1, 3, figsize=(9, 3), tight_layout=True)
ax = ax.ravel()

sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue=adata.obs[f'kmeans_7'], palette='Dark2', alpha=0.7, s=3, ax=ax[0], rasterized=True)
ax[0].set_title('k-means k=7, seed=0')
sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue=adata.obs[f'Agglomerative_euclidean_k7'], palette='Dark2', alpha=0.7, s=3, ax=ax[1], rasterized=True)
# compute the adjusted rand index between k-means and agglomerative clustering
ari = adjusted_rand_score(adata.obs[f'kmeans_7'], adata.obs[f'Agglomerative_euclidean_k7'])
ax[1].set_title('Agglomerative k=7\nvs k-means ARI={:.2f}'.format(ari))
sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue=adata.obs[f'hdbscan_min10_euclidean'], palette='Dark2', alpha=0.7, s=3, ax=ax[2], rasterized=True)
ari = adjusted_rand_score(adata.obs[f'kmeans_7'], adata.obs[f'hdbscan_min10_euclidean'])
ax[2].set_title('HDBSCAN metric=euclidean\nvs k-means ARI={:.2f}'.format(ari))

for i in range(3):
    ax[i].set_xlabel('PC1')
    ax[i].set_ylabel('PC2')
    sns.despine(ax=ax[i])
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].legend(title='Cluster', markerscale=3)

fig.savefig('figures/fig2/clustering_method_comparison.pdf', bbox_inches='tight', dpi=300)

plt.show()


```
