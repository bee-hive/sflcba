# sflcba/utils.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_embedding(adata, color_key, title="SIFT Embedding"):
    """
    Plot a 2D embedding (e.g., PCA) from an AnnData object.
    """
    plt.figure(figsize=(4, 4))
    sns.scatterplot(x=adata.obsm['X_pca'][:,0], y=adata.obsm['X_pca'][:,1],
                    hue=adata.obs[color_key], s=1, alpha=0.7, palette='viridis')
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title=color_key)
    sns.despine()
    plt.show()
