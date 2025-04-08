# sflcba/cluster.py
from sklearn.cluster import KMeans

def cluster_sift_embedding(adata, k_values=range(3, 11)):
    """
    Given an AnnData object with SIFT embeddings in .X and metadata in .obs,
    perform k-means clustering for each k in k_values.
    The cluster labels and the within-cluster sum of squares (wccs) for each k
    are added to the AnnData object.
    """
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(adata.X)
        label_key = f'kmeans_{k}'
        adata.obs[label_key] = kmeans.labels_
        adata.obs[label_key] = adata.obs[label_key].astype('category')
        if not hasattr(adata, 'uns'):
            adata.uns = {}
        adata.uns[label_key] = {'wccs': kmeans.inertia_}
    return adata
