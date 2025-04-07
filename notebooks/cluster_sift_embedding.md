---
jupyter:
  jupytext:
    formats: ipynb,md
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

### Perform k-means clustering of the anndata object

Splitting this up into a separate notebook since it takes longer to run compared to other downstream analysis

```python
import numpy as np
import random
import anndata as ad
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(0)
np.random.seed(0)
```

```python
# load the anndata file with the SIFT descriptors
filename = '/gladstone/engelhardt/lab/adamw/saft_figuren/analysis/adata_20250225.h5ad'
adata = ad.read_h5ad(filename)
adata
```

### Perform K-means clustering of the entire dataset

Test with `k=3`, `k=4`, ..., `k=8`. Select the optimal number of clusters by saving the WCCS and silhouette scores for each `k`.

```python
# k-means clustering of SIFT descriptors
k_values = np.arange(3, 11)

for k in k_values:
    print(k)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(adata.X)
    adata.obs['kmeans_{}'.format(k)] = kmeans.labels_
    adata.obs['kmeans_{}'.format(k)] = adata.obs['kmeans_{}'.format(k)].astype('category')
    # add the k-means inertia (WCCS) score and silhouette score as adata.uns object under the key 'kmeans_{}'.format(k)
    adata.uns['kmeans_{}'.format(k)] = {'wccs': kmeans.inertia_}

adata
```

```python
# plot the k-means clustering WCCS scores as a function of the number of clusters (k)
fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True)
wccs_scores = [adata.uns['kmeans_{}'.format(k)]['wccs'] for k in k_values]
ax.plot(k_values, wccs_scores)
ax.set_xlabel('k')
ax.set_ylabel('WCCS')
sns.despine(ax=ax)
plt.show()
```

```python
# save the anndata file with the k-means clustering results
filename = filename.replace('.h5ad', '_kmeans.h5ad')
adata.write(filename)
```

```python

```
