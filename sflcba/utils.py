# sflcba/utils.py
import re
import matplotlib.pyplot as plt
import seaborn as sns
import tiffile as tiff

def sorted_nicely(l):
    """Sort the given iterable in human order."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

def load_binary_image(image_path, threshold=3.5):
    """
    Load an image from the given path and convert it to a binary format according to the specified threshold.
    """
    # Assuming the image is in grayscale format
    image = tiff.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    # Convert the image to binary format
    image = image > threshold
    
    return image

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
