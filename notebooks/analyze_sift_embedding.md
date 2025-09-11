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

### Given an embedding of SIFT descriptors and their corresponding metadata, perform downstream analysis

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
from sklearn.metrics import silhouette_score
import scipy.stats as stats
from sflcba.stats import cluster_enrichment

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

### Convert time and distance units from frames and pixels to hours and um

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
# plot the shape of the PCA embeddings after dropping NaN values
adata.obsm['X_pca'][:, 0]
```

```python
# move PC1 and PC2 from adata.obsm['X_pca'] to adata.obs['PC1'] and adata.obs['PC2']
adata.obs['PC1'] = adata.obsm['X_pca'][:, 0]
adata.obs['PC2'] = adata.obsm['X_pca'][:, 1]
adata.obs['PC3'] = adata.obsm['X_pca'][:, 2]
adata.obs['PC4'] = adata.obsm['X_pca'][:, 3]
```

```python
# compute a replicate_id integer based on the well_id
# if the well_id ends in an odd number, then the replicate_id is 1
# if the well_id ends in an even number, then the replicate_id is 0
adata.obs['replicate_id'] = adata.obs['well_id'].apply(lambda x: int(x[-1]) % 2)
```

```python
# compute the ROI radius for each SIFT descriptor
adata.obs['roi_radius'] = adata.obs['scales'] * (2 ** (adata.obs['octaves'] + 1))
adata.obs['roi_radius'].value_counts()
```

```python
# make a copy of the entire adata object before subsetting
adata_full = adata.copy()

# subset the entire adata object to just 50k randomly sampled rows
# this should help with runtime issues and overpowered statistical testing
num_rows = 50000
# num_rows = adata.obs.shape[0]
adata = adata[np.random.choice(adata.shape[0], num_rows, replace=False), :]
adata.obs.reset_index(drop=True, inplace=True)
adata
```

```python
adata.obs.head()
```

### Compute some SIFT summary statistics that show the number of keypoints per image

```python
entropy_df = adata_full.obs[['donor_id', 'time', 'well_id', 'replicate_id','rasa2ko_titration', 'et_ratio', 'entropy', 'p_areas', 'n_og_keypoints']].drop_duplicates()
entropy_df
```

```python
entropy_df['n_og_keypoints'].describe()
```

```python
# plot a histogram of the number of keypoints per image
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
entropy_df['n_og_keypoints'].hist(bins=50, ax=ax)
ax.set_xlabel('# SIFT keypoints per image (d$\mathrm{_n}$)')
ax.set_ylabel('Count')
ax.set_title('N={} images'.format(len(entropy_df)))
ax.grid(False)
sns.despine(ax=ax)
fig.savefig('figures/fig2/n_keypoints_hist.pdf', bbox_inches='tight', dpi=300)
plt.show()
```

```python
# make a scatterplot of the number of keypoints vs RFP area with the hue being time
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
sns.scatterplot(ax=ax, data=entropy_df, x='p_areas', y='n_og_keypoints', s=1, alpha=0.5, rasterized=True)
# annotate with seaborn line of best fit
sns.regplot(ax=ax, data=entropy_df, x='p_areas', y='n_og_keypoints', line_kws={'color': 'grey', 'lw': 1, 'ls': '--'}, scatter=False)
# compute the pearson correlation coefficient
result = stats.pearsonr(entropy_df['p_areas'], entropy_df['n_og_keypoints'])
# annotate with the R2 value
ax.text(x=0.8, y=0.65, s=f"r={result.statistic:.2f}", transform=ax.transAxes, ha='center', va='center')
ax.set_xlabel('RFP+ area (μm$^2$)')
ax.set_ylabel('# SIFT keypoints (d$\mathrm{_n}$)')
ax.set_title('N={} images'.format(len(entropy_df)))

sns.despine(ax=ax)
fig.savefig('figures/fig2/n_keypoints_vs_RFP+_scatter.pdf', bbox_inches='tight', dpi=300)
plt.show()
```

```python
result.statistic, result.pvalue
```

```python
np.finfo(np.float64).tiny
```

### Investigate clustering results

Compute silhouette scores for just a subset of the overall dataset since silhouette is O(n^2) in time complexity. Note that the k-means clustering was performed on the entire adata object prior to downsampling to `num_rows` points.

```python
# k-means clustering of SIFT descriptors
k_values = np.arange(3, 11)

# # compute the silhouette score for each value of k on adata
# silhouettes = []
# for k in k_values:
#     colname = 'kmeans_{}'.format(k)
#     # make sure that the clustering results are in category format
#     adata.obs[colname] = adata.obs[colname].astype('category')
#     score = silhouette_score(adata.X, adata.obs[colname])
#     silhouettes.append(score)
```

```python


fig, ax = plt.subplots(1, 2, figsize=(4, 2), tight_layout=True)
ax = ax.flatten()

# # plot the silhouette score for each value of k on adata_B4_donor1
# ax[0].plot(k_values, silhouettes)
# ax[0].set_xlabel('Number of clusters (K)')
# ax[0].set_ylabel('Silhouette score')
# ax[0].set_title('K-means of SIFT matrix')
# sns.despine(ax=ax[0])

# plot the wccs score from adata.uns['kmeans_{k}] for each value of k on the entire dataset
wccs = [ adata.uns['kmeans_{}'.format(k)]['wccs'] for k in k_values ]
ax[1].plot(k_values, wccs)
ax[1].set_xlabel('Number of clusters (K)')
ax[1].set_ylabel('Sum of squared distances\nto cluster centers')
ax[1].set_title('K-means of SIFT matrix')
sns.despine(ax=ax[1])

# draw vertical dashed grey line at k=7 for both subplots
for i in range(2):
    ax[i].axvline(7, color='grey', linestyle='--', alpha=0.5)

# fig.savefig('figures/fig2/kmeans_silhouette_wccs.pdf', bbox_inches='tight')

plt.show()
```

```python
# re-run PCA so we can compute the variance explained by each PC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# normalize the SIFT detectors
scaler = StandardScaler()
X=adata.X
scaler.fit(X)
X=scaler.transform(X)

# run pca using 30 components
pca = PCA(n_components=30)
x_new = pca.fit_transform(X)

# plot the explained variance ratio
fig, ax = plt.subplots(1, 1, figsize=(3, 3), tight_layout=True)
ax.scatter(np.arange(1, len(pca.explained_variance_ratio_)+1, 1), pca.explained_variance_ratio_)
ax.set_xlabel('PC')
ax.set_ylabel('Explained variance ratio')
sns.despine(ax=ax)
# ax.set_xticks(np.arange(1, len(pca.explained_variance_ratio_)+1, 1))
ax.set_title('PCA of SIFT descriptor matrix')

# plut the cumulative explained variance
fig, ax = plt.subplots(1, 1, figsize=(3, 3), tight_layout=True)
ax.scatter(np.arange(1, len(pca.explained_variance_ratio_)+1, 1), np.cumsum(pca.explained_variance_ratio_))
ax.set_xlabel('PC')
ax.set_ylabel('Cumulative explained variance ratio')
sns.despine(ax=ax)
# ax.set_xticks(np.arange(1, len(pca.explained_variance_ratio_)+1, 1))
ax.set_title('PCA of SIFT descriptor matrix')
# set the y-axis to be between 0 and 1
ax.set_ylim(0, 1)

plt.show()
```

```python
# show the PCA embeddings colored by K-means cluster
fig, ax = plt.subplots(4, 4, figsize=(8, 8), tight_layout=True)


# merge the bottom row from four columns into two columns

# create a new gridspec for the bottom left subplot
gs = ax[3, 0].get_gridspec()
for j in range(0, 2):
    ax[3, j].remove()
axbottomleft = fig.add_subplot(gs[3, 0:2])

# create a new gridspec for the bottom right subplot
gs = ax[3, 2].get_gridspec()
for j in range(2, 4):
    ax[3, j].remove()
axbottomright = fig.add_subplot(gs[3, 2:4])

ax = ax.flatten()

for i, k in enumerate(k_values[:8]):
    sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue='kmeans_{}'.format(k), palette='Dark2', alpha=0.7, s=1, ax=ax[i], rasterized=True)
    ax[i].set_xlabel('PC1')
    ax[i].set_ylabel('PC2')
    sns.despine(ax=ax[i])
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_title('K-means with k={}'.format(k))
    if k == 10:
        ax[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Cluster ID', markerscale=5)
    else:
        # remove the legend
        ax[i].legend_.remove()

# plot the PCA embeddings for higher PC dimensions
sns.scatterplot(data=adata.obs, x='PC1', y='PC3', hue='kmeans_7', palette='Dark2', alpha=0.7, s=1, ax=ax[8], rasterized=True)
sns.scatterplot(data=adata.obs, x='PC1', y='PC4', hue='kmeans_7', palette='Dark2', alpha=0.7, s=1, ax=ax[9], rasterized=True)
sns.scatterplot(data=adata.obs, x='PC2', y='PC3', hue='kmeans_7', palette='Dark2', alpha=0.7, s=1, ax=ax[10], rasterized=True)
sns.scatterplot(data=adata.obs, x='PC2', y='PC4', hue='kmeans_7', palette='Dark2', alpha=0.7, s=1, ax=ax[11], rasterized=True)

for i in range(8, 12):
    sns.despine(ax=ax[i])
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_title('K-means with k=7')
    ax[i].legend_.remove()

# plot the explained variance ratio and cumulative explained variance for the top 30 PCs
axbottomleft.scatter(np.arange(1, len(pca.explained_variance_ratio_)+1, 1), pca.explained_variance_ratio_, s=5)
axbottomleft.set_xlabel('PC')
axbottomleft.set_ylabel('Explained variance ratio')
sns.despine(ax=axbottomleft)
axbottomleft.set_title('PCA of SIFT descriptor matrix')

# plut the cumulative explained variance
axbottomright.scatter(np.arange(1, len(pca.explained_variance_ratio_)+1, 1), np.cumsum(pca.explained_variance_ratio_), s=5)
axbottomright.set_xlabel('PC')
axbottomright.set_ylabel('Cumulative explained variance')
sns.despine(ax=axbottomright)
# ax.set_xticks(np.arange(1, len(pca.explained_variance_ratio_)+1, 1))
axbottomright.set_title('PCA of SIFT descriptor matrix')
# set the y-axis to be between 0 and 1
axbottomright.set_ylim(0, 1)

# fig.savefig('figures/fig2/kmeans_pca_embeddings.pdf', bbox_inches='tight', dpi=300)

plt.show()
```

Despite the silhouette score being highest for `k=4` clusters, we chose to move forward with `k=7` clusters for downstream analysis as this gave us additional granularity for explaining variation in SIFT descriptors.


### Show the clustering results alongside ROIs for representative descriptors

This will help us qualitatively describe the meaning of each cluster

```python
def plot_sift_roi(row, ax, offset=None, cluster_col='kmeans_7', add_rfp_mask=False, um_per_pixel=4.975):
    '''
    Plot the SIFT descriptors for a single ROI. Input is a single row of the adata.obs dataframe that contains the columns: filename, x, y,
    '''
    image = tiff.imread(row['filename'])
    # normalize the intensity of the image to range [0, 1]
    image = (image - image.min()) / (image.max() - image.min())

    # compute the radius for this SIFT descriptor
    radius = row['roi_radius']

    # only show the true ROI if offset is None, otherwise show the ROI with the specified offset
    if offset is None:
        offset = radius
    
    # crop the image to the ROI
    xmin, xmax = row['x'] - offset, row['x'] + offset
    ymin, ymax = row['y'] - offset, row['y'] + offset
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > image.shape[0]:
        xmax = image.shape[0]
    if ymax > image.shape[1]:
        ymax = image.shape[1]
    image = image[xmin:xmax, ymin:ymax]

    # optional argument to load the red channel where RFP is expressed in cancer cells
    if add_rfp_mask:
        rfp_filename = row['filename'].replace('phase_registered', 'red_registered')
        rfp_image = tiff.imread(rfp_filename)

        # # normalize the entire image intensity values to range [0, 1]
        # rfp_image = (rfp_image - rfp_image.min()) / (rfp_image.max() - rfp_image.min())

        rfp_image = rfp_image[xmin:xmax, ymin:ymax]
        # threshold the RFP channel into a binary mask
        rfp_mask = rfp_image > 3.5
        ax.imshow(rfp_mask, cmap='Reds', alpha=1.0, vmin=0, vmax=1)
        phase_alpha = 0.7
    else:
        phase_alpha = 1.0
    
    # plot the greyscale image
    ax.imshow(image, cmap='gray', alpha=phase_alpha, vmin=0, vmax=1)

    # find the center point of the image.shape matrix
    height, width = image.shape
    x = (width - 1) / 2.0
    y = (height - 1) / 2.0
    
    # set color of title and circle based on the cluster_col value and the Dark2 colormap
    color = cm.Dark2(row[cluster_col])
    # annotate the ROI with a circle that has r=radius
    circle = patches.Circle((x, y), radius, edgecolor=color, facecolor='none')
    ax.add_patch(circle)

    # add a line going from the center of the circle to the perimeter of the circle and an angle of row['orientations']
    angle = row['orientations']
    ax.plot([x, x + radius * np.cos(angle)], [y, y + radius * np.sin(angle)], color=color)


    sns.despine(ax=ax)
    ax.set_title('cluster={}\nRASA2KO={}, E:T={}\ndonor={}, well={}, t={}'.format(row[cluster_col], round(row['rasa2ko_titration'], 2), round(row['et_ratio'], 2), row['donor_id'], row['well_id'], row['time']), fontsize=6)
    # set color of the title based on the cluster_col value and the Dark2 colormap
    ax.title.set_color(color)

    # multiply all the x and y ticklabels by um_per_pixel
    # Get current x-tick locations
    original_xticks = ax.get_xticks()
    # Calculate new tick labels by multiplying by 2
    new_xticklabels = [f"{tick * um_per_pixel:.0f}" for tick in original_xticks] # Format as integers
    # Set the new tick labels
    ax.set_xticklabels(new_xticklabels)
    # get current y-tick locations
    original_yticks = ax.get_yticks()
    new_yticklabels = [f"{tick * um_per_pixel:.0f}" for tick in original_yticks] # Format as integers
    ax.set_yticklabels(new_yticklabels)

fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
plot_sift_roi(adata.obs.iloc[0], ax, offset=15, add_rfp_mask=True, um_per_pixel=um_per_pixel)
```

```python
def plot_two_random_rois(adata, cluster_id, ax1, ax2, cluster_col='kmeans_7', offset=15, add_rfp_mask=False, um_per_pixel=um_per_pixel):
    ''' Plot two random ROIs belonging to the cluster_col==cluster_id '''
    temp_df = adata.obs[adata.obs[cluster_col]==cluster_id]
    # temp_df = temp_df.iloc[np.random.choice(temp_df.shape[0], 2, replace=False), :]
    plot_sift_roi(temp_df.iloc[2], ax1, offset=offset, add_rfp_mask=add_rfp_mask, um_per_pixel=um_per_pixel)
    plot_sift_roi(temp_df.iloc[3], ax2, offset=offset, add_rfp_mask=add_rfp_mask, um_per_pixel=um_per_pixel)

# create a 7x3 grid of subplots where the middle 3 rows and columns are merged into one 3x3 subplot 
fig, ax = plt.subplots(3, 6, figsize=(8.5, 4.25), tight_layout=True)
# merge the middle 3 rows and columns into one 3x3 subplot
gs = ax[0, 2].get_gridspec()
for i in range(0, 2):
    for j in range(2, 4):
        ax[i, j].remove()

# plot the PCA embedding for donor_id==1 and well_id==B4 in the big center subplot
axbig = fig.add_subplot(gs[0:2, 2:4])
sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue='kmeans_7', palette='Dark2', alpha=1, s=2, ax=axbig, rasterized=True)
axbig.set_title('SIFT embedding (D={})'.format(num_rows))
# axbig.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='K-means k=7')
axbig.legend(title='Cluster ID', markerscale=4)
sns.despine(ax=axbig)
axbig.set_xticks([])
axbig.set_yticks([])

# plot ROIs on the left and right of the big center subplot
plot_two_random_rois(adata, 5, ax[0, 0], ax[0, 1], add_rfp_mask=True, um_per_pixel=um_per_pixel)
plot_two_random_rois(adata, 6, ax[1, 0], ax[1, 1], add_rfp_mask=True, um_per_pixel=um_per_pixel)
plot_two_random_rois(adata, 4, ax[2, 0], ax[2, 1], add_rfp_mask=True, um_per_pixel=um_per_pixel)
plot_two_random_rois(adata, 2, ax[0, 4], ax[0, 5], add_rfp_mask=True, um_per_pixel=um_per_pixel)
plot_two_random_rois(adata, 0, ax[1, 4], ax[1, 5], add_rfp_mask=True, um_per_pixel=um_per_pixel)
plot_two_random_rois(adata, 3, ax[2, 4], ax[2, 5], add_rfp_mask=True, um_per_pixel=um_per_pixel)
plot_two_random_rois(adata, 1, ax[2, 2], ax[2, 3], add_rfp_mask=True, um_per_pixel=um_per_pixel)

fig.savefig('figures/fig2/multipanel_embedding_rois_rfp.pdf', bbox_inches='tight', dpi=300)

plt.show()
```

```python
# create a 7x3 grid of subplots where the middle 3 rows and columns are merged into one 3x3 subplot 
fig, ax = plt.subplots(3, 6, figsize=(8.5, 4.25), tight_layout=True)
# merge the middle 3 rows and columns into one 3x3 subplot
gs = ax[0, 2].get_gridspec()
for i in range(0, 2):
    for j in range(2, 4):
        ax[i, j].remove()

# plot the PCA embedding for donor_id==1 and well_id==B4 in the big center subplot
axbig = fig.add_subplot(gs[0:2, 2:4])
sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue='kmeans_7', palette='Dark2', alpha=1, s=2, ax=axbig, rasterized=True)
axbig.set_title('SIFT embedding (D={})'.format(num_rows))
# axbig.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='K-means k=7')
axbig.legend(title='Cluster ID', markerscale=4)
sns.despine(ax=axbig)
axbig.set_xticks([])
axbig.set_yticks([])

# plot ROIs on the left and right of the big center subplot
plot_two_random_rois(adata, 5, ax[0, 0], ax[0, 1], add_rfp_mask=False, um_per_pixel=um_per_pixel)
plot_two_random_rois(adata, 6, ax[1, 0], ax[1, 1], add_rfp_mask=False, um_per_pixel=um_per_pixel)
plot_two_random_rois(adata, 4, ax[2, 0], ax[2, 1], add_rfp_mask=False, um_per_pixel=um_per_pixel)
plot_two_random_rois(adata, 2, ax[0, 4], ax[0, 5], add_rfp_mask=False, um_per_pixel=um_per_pixel)
plot_two_random_rois(adata, 0, ax[1, 4], ax[1, 5], add_rfp_mask=False, um_per_pixel=um_per_pixel)
plot_two_random_rois(adata, 3, ax[2, 4], ax[2, 5], add_rfp_mask=False, um_per_pixel=um_per_pixel)
plot_two_random_rois(adata, 1, ax[2, 2], ax[2, 3], add_rfp_mask=False, um_per_pixel=um_per_pixel)

fig.savefig('figures/fig2/multipanel_embedding_rois_phase.pdf', bbox_inches='tight', dpi=300)


plt.show()
```

### Plot large ROIs and annotate them with multiple SIFT keypoints colored by K-means cluster ID

```python
def plot_large_roi_w_keypoints(df, ax, well_id, donor_id, time_point, roi_center, roi_radius, um_per_pixel=4.975):
    '''
    Given a specific image, ROI within that image, and full set of SIFT descriptors for that ROI, plot the SIFT keypoints on top of the bright field + RFP image

    Parameters
    ----------
     df : pd.DataFrame
        dataframe of SIFT keypoints and descriptors. Columns should include ['well_id', 'donor_id', 'time', 'filename', 'x', 'y', 'scales', 'octaves', 'orientations', 'kmeans_7']
    ax : matplotlib axes object
        axes object to plot on
    well_id : str
        well id of the image we wish to plot
    donor_id : str
        donor id of the image we wish to plot
    time_point : int
        time point of the image we wish to plot
    roi_center : tuple of int
        center of the ROI in x-y coordinates
    roi_radius : int
        radius of the ROI in pixels
    um_per_pixel : float
        number of microns per pixel
    '''
    # subset the dataframe to the specified well_id, donor_id, and time
    df = df[(df['well_id']==well_id) & (df['donor_id']==donor_id) & (df['time']==time_point)]

    # create a list of all the image file paths that correspond to this donor_id
    phase_filename = df['filename'].iloc[0]
    red_filename = phase_filename.replace('phase', 'red')

    # subset the list to the specified time point and load the images
    resized_latish_phase = tiff.imread(phase_filename)
    resized_latish_red = tiff.imread(red_filename)
    # threshold the red channel into a binary mask
    red_frame = resized_latish_red > 3.5

    # normalize intensity of the phase image
    phase_frame = cv.normalize(resized_latish_phase, None, 0, 255, cv.NORM_MINMAX).astype('uint8')

    # find the x_min, x_max, y_min, y_max boundaries of the image based on the roi_center and roi_radius
    x_min = roi_center[0] - roi_radius
    x_max = roi_center[0] + roi_radius
    y_min = roi_center[1] - roi_radius
    y_max = roi_center[1] + roi_radius

    ax.imshow(red_frame[x_min:x_max, y_min:y_max], cmap='Reds', alpha = 1.0)
    ax.imshow(phase_frame[x_min:x_max, y_min:y_max], cmap='gray', alpha = .75)

    # find all the SIFT keypoints in this image (each row in df) and annotate their location
    for idx, row in df.iterrows():
        # check to see if the x-y position of this SIFT keypoint falls within xx_min:x_max, y_min:y_max
        key_x = row['x']
        key_y = row['y']
        if (key_x < x_max) and (key_x > x_min) and (key_y < y_max) and (key_y > y_min):
            key_radius = row['roi_radius']
            # set color of title and circle based on the cluster_col value and the Dark2 colormap
            color = cm.Dark2(row['kmeans_7'])
            # annotate the ROI with a circle that has r=radius
            circle = patches.Circle((key_x-x_min, key_y-y_min), key_radius, edgecolor=color, facecolor='none')
            ax.add_patch(circle)
        
            # add a line going from the center of the circle to the perimeter of the circle and an angle of row['orientations']
            angle = row['orientations']
            ax.plot([key_x-x_min, key_x-x_min + key_radius * np.cos(angle)], [key_y-y_min, key_y-y_min + key_radius * np.sin(angle)], color=color)

    ax.set_title('Well: {}, Donor: {}, Time: {}'.format(well_id,donor_id,time_point), size = 12)
    # remove the axes ticks and labels
    sns.despine(ax=ax)
    ax.set_xlim([0, roi_radius*2])
    ax.set_ylim([roi_radius*2, 0])

    # offset the x and y tick labels by the ROI center values
    xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    ytick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    ax.set_xticklabels([str(int((int(label) + roi_center[0]) * um_per_pixel)) for label in xtick_labels])
    ax.set_yticklabels([str(int((int(label) + roi_center[1]) * um_per_pixel)) for label in ytick_labels])

    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    
    return ax

fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
ax = ax.flatten()
plot_large_roi_w_keypoints(adata_full.obs, ax[0], well_id='H9', donor_id=1, time_point=10, roi_center=(500,500), roi_radius=200, um_per_pixel=um_per_pixel)
plot_large_roi_w_keypoints(adata_full.obs, ax[1], well_id='H9', donor_id=1, time_point=110, roi_center=(500,500), roi_radius=200, um_per_pixel=um_per_pixel)
plot_large_roi_w_keypoints(adata_full.obs, ax[2], well_id='B5', donor_id=1, time_point=10, roi_center=(300,300), roi_radius=200, um_per_pixel=um_per_pixel)
plot_large_roi_w_keypoints(adata_full.obs, ax[3], well_id='B5', donor_id=1, time_point=110, roi_center=(300,300), roi_radius=200, um_per_pixel=um_per_pixel)

fig.savefig('figures/fig2/large_roi_w_keypoints.pdf', bbox_inches='tight', dpi=300)

plt.show()
```

### Plot the entire well for representative image frames

Save examples both with and without the RFP channel superimposed

```python
def plot_entire_well(df, ax, well_id, donor_id, time_point, trim=100, rfp=False):
    '''
    Plot the entire brightfield (and optionally RFP) image for a single well.

    Parameters
    ----------
     df : pd.DataFrame
        dataframe of SIFT keypoints and descriptors. Columns should include ['well_id', 'donor_id', 'time', 'filename', 'x', 'y', 'scales', 'octaves', 'orientations', 'kmeans_7']
    ax : matplotlib axes object
        axes object to plot on
    well_id : str
        well id of the image we wish to plot
    donor_id : str
        donor id of the image we wish to plot
    time_point : int
        time point of the image we wish to plot
    trim : int
        number of pixels to trim from the edges of the image
    rfp : bool
        whether or not to plot the red channel
    '''
    # subset the dataframe to the specified well_id, donor_id, and time
    df = df[(df['well_id']==well_id) & (df['donor_id']==donor_id) & (df['time']==time_point)]

    # create a list of all the image file paths that correspond to this donor_id
    phase_filename = df['filename'].iloc[0]
    red_filename = phase_filename.replace('phase', 'red')

    # subset the list to the specified time point and load the images
    resized_latish_phase = tiff.imread(phase_filename)
    resized_latish_red = tiff.imread(red_filename)
    # threshold the red channel into a binary mask
    red_frame = resized_latish_red > 3.5

    # normalize intensity of the phase image
    phase_frame = cv.normalize(resized_latish_phase, None, 0, 255, cv.NORM_MINMAX).astype('uint8')

    # plot the phase image with the red mask superimposed
    trim=100
    if rfp:
        ax.imshow(red_frame[trim:-trim,trim:-trim], cmap='Reds', alpha = 1.0)
    ax.imshow(phase_frame[trim:-trim,trim:-trim], cmap='gray', alpha = .75)
    ax.set_title('Well: {}, Donor: {}, Time: {}'.format(well_id,donor_id,time_point), size = 12)
    # remove the axes ticks and labels
    ax.axis('off')


```

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
ax = ax.flatten()
plot_entire_well(adata_full.obs, ax[0], well_id='B4', donor_id=1, time_point=90, trim=100, rfp=False)
plot_entire_well(adata_full.obs, ax[1], well_id='B4', donor_id=1, time_point=90, trim=100, rfp=True)
fig.savefig('figures/B4_donor1_time45.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

### Analyze ROIs surrounding each SIFT keypoint to quantitatively describes K-means clusters obtained from the SIFT descriptor matrix

This can include looking at position of keypoints in the well, RFP intensity, spatial heterogeneity of phase and red images surrounding each ROI.


Group clusters into their ontological categories when plotting violins and boxplots

```python
# add a new column to adata.obs that is the cluster_nickname
cluster_nickname_dict = {
    0: 'mixed',
    1: 'singlets_1',
    2: 'edges_1',
    3: 'edges_2',
    4: 'singlets_2',
    5: 'aggregates_1',
    6: 'aggregates_2'
}
cluster_group_dict = {
    0: 'mixed',
    1: 'singlets',
    2: 'edges',
    3: 'edges',
    4: 'singlets',
    5: 'aggregates',
    6: 'aggregates'
}
adata.obs['cluster_nickname'] = adata.obs['kmeans_7'].map(cluster_nickname_dict)
adata.obs['cluster_group'] = adata.obs['kmeans_7'].map(cluster_group_dict)

# create a column for the dark2 color palette mapping of each cluster
cmap = plt.cm.get_cmap('Dark2')
colors = cmap(np.arange(7))
cluster_color_dict = {
    0: colors[0],
    1: colors[1],
    2: colors[2],
    3: colors[3],
    4: colors[4],
    5: colors[5],
    6: colors[6]
}
```

### Start by looking at the distance of each SIFT keypiont from the center of the well

Keypoints belonging to edge clusters will have a larger distance from the center and a smaller standard deviation in distance compared to non-edge clusters.

```python
# use the xy-coordinates of each SIFT descriptor to find its radius (in pixels) away from the center of the image
# start by shifting x and y coordinates to be centered at 0,0
adata.obs['x_centered'] = adata.obs['x_um'] - adata.obs['x_um'].mean()
adata.obs['y_centered'] = adata.obs['y_um'] - adata.obs['y_um'].mean()
# compute the radius of each SIFT descriptor
adata.obs['distance_from_center'] = np.sqrt(adata.obs['x_centered']**2 + adata.obs['y_centered']**2)
adata.obs['distance_from_center'].describe()

# compute square root of the distance_from_center values
adata.obs['sqrt_distance_from_center'] = np.sqrt(adata.obs['distance_from_center']) / adata.obs['distance_from_center'].max()
```

```python
# compute the standard deviation in the distance_from_center values within each kmeans_7 cluster
cluster_positions_df = []
for k, group in adata.obs.groupby('kmeans_7'):
    mean = group['distance_from_center'].mean()
    stdev = group['distance_from_center'].std()
    cluster_group = group['cluster_group'].iloc[0]
    cluster_positions_df.append(pd.DataFrame({'kmeans_7': [k], 
                                          'cluster_group': [cluster_group],
                                          'distance_from_center_mean': [mean],
                                          'distance_from_center_stdev': [stdev]}))
cluster_positions_df = pd.concat(cluster_positions_df, ignore_index=True)
cluster_positions_df
```

```python
# compute the standard deviation in the distance_from_center values within each cluster group
cluster_group_positions_df = []
for g, group in adata.obs.groupby('cluster_group'):
    mean = group['distance_from_center'].mean()
    stdev = group['distance_from_center'].std()
    cluster_group = group['cluster_group'].iloc[0]
    cluster_group_positions_df.append(pd.DataFrame({ 
                                          'cluster_group': [g],
                                          'distance_from_center_mean': [mean],
                                          'distance_from_center_stdev': [stdev]}))
cluster_group_positions_df = pd.concat(cluster_group_positions_df, ignore_index=True)
cluster_group_positions_df
```

```python
# plot the SIFT descriptor radii from image centers for each kmeans_7 cluster
fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
ax = ax.flatten()

# violin plot of the radius from center of each SIFT descriptor
sns.violinplot(ax=ax[0], data=adata.obs, x='kmeans_7', y='distance_from_center', hue='kmeans_7', legend=False, palette='Dark2')
ax[0].set_title('SIFT keypoint position in well')
ax[0].set_xlabel('K-means cluster ID')
ax[0].set_ylabel('Distance from center of well (μm)')
sns.despine(ax=ax[0])

# bar plot of the variance of the radius from center
sns.barplot(ax=ax[1], data=cluster_positions_df, x='kmeans_7', y='distance_from_center_stdev', hue='kmeans_7', legend=False, palette='Dark2')
ax[1].set_title('SIFT keypoint position standard deviation per cluster')
ax[1].set_xlabel('K-means cluster ID')
ax[1].set_ylabel('Standard deviation of distance from center of well (μm)')
sns.despine(ax=ax[1])
plt.show()
```

```python
edge_df = adata.obs[adata.obs['cluster_group'] == 'edges']
non_edge_df = adata.obs[adata.obs['cluster_group'] != 'edges']

edge_df['distance_from_center'].describe()
```

```python
non_edge_df['distance_from_center'].describe()
```

```python
# divide the standard deviation of the non_edge_df['distance_from_center'] by the standard deviation of the edge_df['distance_from_center']
edge_stdev = edge_df['distance_from_center'].std()
non_edge_stdev = non_edge_df['distance_from_center'].std()
non_edge_stdev / edge_stdev
```

```python
# compute a t-test comparing the standard deviation of the non_edge_df['distance_from_center'] to the standard deviation of the edge_df['distance_from_center']
stats.ttest_ind(non_edge_df['distance_from_center'], edge_df['distance_from_center'], equal_var=False)
```

```python
# convert the roi_radius column to units of um
adata.obs['roi_radius_um'] = adata.obs['roi_radius'] * um_per_pixel
```

```python
# plot the PCA embedding annotated by roi_radius
fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue='roi_radius_um', palette='coolwarm', ax=ax, s=1, alpha=0.7)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('SIFT embedding (D={})'.format(num_rows))
ax.legend(title='ROI radius (μm)', markerscale=5)
sns.despine(ax=ax)
ax.set_xticks([])
ax.set_yticks([])
plt.show()
```

```python
# plot a histogram of roi_radius
fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
sns.histplot(data=adata.obs, x='roi_radius_um', bins=20, ax=ax)
ax.set_xlabel('ROI radius (μm)')
ax.set_ylabel('Count')
ax.set_title('Cancer cell abundance around SIFT keypoints'.format(num_rows))
sns.despine(ax=ax)
plt.show()
```

```python
# say that all plate-wide striations have an ROI radius of >250 pixels
adata.obs['plate_wide_striation'] = adata.obs['roi_radius'] > 250
```

```python
# find the proportion of rows in cluster 0 that are plate-wide striations
num_cluster_0_rows = adata.obs[adata.obs['kmeans_7'] == 0].shape[0]
plate_wide_striation_frac = adata.obs[adata.obs['kmeans_7'] == 0]['plate_wide_striation'].mean()
plate_wide_striation_frac, num_cluster_0_rows, num_cluster_0_rows * plate_wide_striation_frac
```

```python
# find the proportion of rows outside of cluster 0 that are plate-wide striations
num_non_cluster_0_rows = adata.obs[adata.obs['kmeans_7'] != 0].shape[0]
plate_wide_striation_frac = adata.obs[adata.obs['kmeans_7'] != 0]['plate_wide_striation'].mean()
plate_wide_striation_frac, num_non_cluster_0_rows, num_non_cluster_0_rows * plate_wide_striation_frac
```

```python
# find the min, median, and max of roi_radius for rows outside of cluster 0
adata.obs[adata.obs['kmeans_7'] != 0]['roi_radius'].describe()
```

### Look at RFP intensity values near each SIFT descriptor

```python
# create a violinplot of roi_mean_rfp_intensity by kmeans_7
fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
sns.violinplot(data=adata.obs, x='kmeans_7', y='roi_mean_rfp_intensity', hue='kmeans_7', palette='Dark2', legend=False, ax=ax)
ax.set_xlabel('K-means cluster ID')
ax.set_ylabel('Mean RFP intensity')
ax.set_title('Cancer cell abundance around SIFT keypoints'.format(num_rows))
sns.despine(ax=ax)
plt.show()
```

```python
# plot the PCA embedding and violin plot of the ROI fraction of RFP+ pixels (roi_rfp_pos_frac)
fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue='roi_rfp_pos_frac', palette='coolwarm', ax=ax[0], s=1, alpha=0.7)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[0].set_title('SIFT embedding (D={})'.format(num_rows))
ax[0].legend(title='ROI RFP+\npixel fraction', markerscale=5)
sns.despine(ax=ax[0])
ax[0].set_xticks([])
ax[0].set_yticks([])

sns.violinplot(data=adata.obs, x='kmeans_7', y='roi_rfp_pos_frac', hue='kmeans_7', palette='Dark2', legend=False, ax=ax[1])
ax[1].set_xlabel('K-means cluster ID')
ax[1].set_ylabel('Fraction of RFP+ pixels')
ax[1].set_title('RFP+ pixel fraction around SIFT keypoints'.format(num_rows))
sns.despine(ax=ax[1])
plt.show()
```

```python
# scatterplot of RFP+ pixel fraction vs roi_mean_rfp_intensity
fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
sns.regplot(data=adata.obs, x='roi_mean_rfp_intensity', y='roi_rfp_pos_frac', ax=ax, scatter_kws={'s': 1, 'alpha': 0.7, 'color': 'k'}, line_kws={'color': 'r', 'lw': 2})
ax.set_xlabel('Mean RFP intensity')
ax.set_ylabel('Fraction of RFP+ pixels')
ax.set_ylim([0, 1])
sns.despine(ax=ax)
plt.show()
```

```python
# plot the PCA embedding annotated by roi_radius
fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)

sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue='roi_radius_um', palette='coolwarm', ax=ax[0], s=1, alpha=0.7, rasterized=True)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[0].set_title('SIFT embedding (D={})'.format(num_rows))
ax[0].legend(title='ROI radius (μm)', markerscale=5)
sns.despine(ax=ax[0])
ax[0].set_xticks([])
ax[0].set_yticks([])

sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue='roi_rfp_pos_frac', palette='coolwarm', ax=ax[1], s=1, alpha=0.7, rasterized=True)
ax[1].set_xlabel('PC1')
ax[1].set_ylabel('PC2')
ax[1].set_title('SIFT embedding (D={})'.format(num_rows))
ax[1].legend(title='ROI RFP+\npixel fraction', markerscale=5)
sns.despine(ax=ax[1])
ax[1].set_xticks([])
ax[1].set_yticks([])

fig.savefig('figures/fig3/pca_embedding_roi_radius_rfp_frac.pdf', bbox_inches='tight', dpi=300)

plt.show()
```

### Look at the Moran's I statistics between clusters to assess the spatial autocorrelation of both the red and phase channels

```python
# plot the PCA embedding and violin plot of the RFP Moran's I (roi_rfp_morans_I)
fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue='roi_rfp_morans_I', palette='coolwarm', ax=ax[0], s=1, alpha=0.7)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[0].set_title('SIFT embedding (D={})'.format(num_rows))
sns.despine(ax=ax[0])
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].legend(title='RFP Moran\'s I', markerscale=5)

sns.violinplot(data=adata.obs, x='kmeans_7', y='roi_rfp_morans_I', hue='kmeans_7', palette='Dark2', legend=False, ax=ax[1])
ax[1].set_xlabel('K-means cluster ID')
ax[1].set_ylabel('RFP Moran\'s I')
ax[1].set_title('RFP Moran\'s I around SIFT keypoints')
sns.despine(ax=ax[1])
plt.show()
```

```python
# plot the PCA embedding and violin plot of the ROI brightfield Moran's I (roi_bf_morans_I)
fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue='roi_bf_morans_I', palette='coolwarm', ax=ax[0], s=1, alpha=0.7)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[0].set_title('SIFT embedding (D={})'.format(num_rows))
sns.despine(ax=ax[0])
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].legend(title='BF Moran\'s I', markerscale=5)

sns.violinplot(data=adata.obs, x='kmeans_7', y='roi_bf_morans_I', hue='kmeans_7', palette='Dark2', legend=False, ax=ax[1])
ax[1].set_xlabel('K-means cluster ID')
ax[1].set_ylabel('BF Moran\'s I')
ax[1].set_title('BF Moran\'s I around SIFT keypoints')
sns.despine(ax=ax[1])
plt.show()
```

Try to tease out if there are any cluster-specific Moran's I deviations within low, med, or high RFP+ ROIs.

```python
# group the roi_rfp_pos_frac into groups based on having a values in the range of 0 to 0.25, 0.25 to 0.5, 0.5 to 0.75, and 0.75 to 1
adata.obs['roi_rfp_pos_frac_quartile'] = pd.cut(adata.obs['roi_rfp_pos_frac'], bins=[-0.01, 0.25, 0.5, 0.75, 1.01], labels=[0, 1, 2, 3])
```

```python
# subset adata.obs to just the rows from clusters 1 and 4
df_polar = adata.obs[(adata.obs['kmeans_7'] == 1) | (adata.obs['kmeans_7'] == 4)].copy()

# do the same for aggregate clusters 5 and 6
df_aggregate = adata.obs[(adata.obs['kmeans_7'] == 5) | (adata.obs['kmeans_7'] == 6)].copy()
```

```python


fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
ax = ax.flatten()

# plot a boxplot of ROI RFP+ fraction of pixels (roi_rfp_pos_frac_quartile) by ROI RFP Moran's I (roi_rfp_morans_I)
sns.boxplot(data=df_polar, x='roi_rfp_pos_frac_quartile', y='roi_rfp_morans_I', hue='kmeans_7', palette='Dark2', ax=ax[0], legend=False)
ax[0].set_xlabel('ROI RFP+ fraction ')
ax[0].set_ylabel('ROI RFP Moran\'s I')
# rename x-tick labels based on quartile values
ax[0].set_xticklabels(['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1'])
sns.despine(ax=ax[0])

# plot the same boxplot as above but have the y-axis be roi_bf_morans_I instead of roi_rfp_morans_I
sns.boxplot(data=df_polar, x='roi_rfp_pos_frac_quartile', y='roi_bf_morans_I', hue='kmeans_7', palette='Dark2', ax=ax[1], legend=False)
ax[1].set_xlabel('ROI RFP+ fraction ')
ax[1].set_ylabel('ROI BF Moran\'s I')
# rename x-tick labels based on quartile values
ax[1].set_xticklabels(['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1'])
sns.despine(ax=ax[1])

plt.show()
```

```python

def boxplots_by_cluster_group(adata, cluster_color_dict=cluster_color_dict, ycol='roi_rfp_pos_frac', huecol='kmeans_7', ylabel='ROI RFP+ fraction', title=None, legend=True):
    fig, ax = plt.subplots(1, 3, figsize=(4, 4), tight_layout=True, sharey=True)

    cluster_groups = ['edges', 'singlets', 'aggregates']

    for i, group in enumerate(cluster_groups):
        plot_data = adata.obs[adata.obs['cluster_group'] == group]
        plot_data['kmeans_7'] = plot_data['kmeans_7'].astype('int')

        # compute a t-test between the two kmeans_7 clusters present in this cluster group
        cluster_ids = plot_data[huecol].unique()
        ttest = stats.ttest_ind(plot_data.loc[plot_data[huecol] == cluster_ids[0],ycol], plot_data.loc[plot_data[huecol] == cluster_ids[1],ycol])

        sns.boxplot(data=plot_data, x='cluster_group', y=ycol, hue=huecol, palette=cluster_color_dict, legend=False, ax=ax[i], fliersize=1)

        # add the p-value to the plot
        p_adj = ttest.pvalue * len(cluster_groups)
        ax[i].text(0.5, 0.99, f'p={p_adj:.1e}', transform=ax[i].transAxes, ha='center', va='top', fontsize=8)
        print('ttest for ', group, ': ', ttest)

        ax[i].set_xlabel('')
        ax[i].set_ylabel(ylabel)
        sns.despine(ax=ax[i])

    ax[1].set_title(title)
    ax[1].set_xlabel('Cluster')
    if legend:
        # create legend for ax[2] that shows the color corresponding to each kmeans_7 value
        handles = []
        for i in range(7):
            handles.append(patches.Patch(color=cluster_color_dict[i], label=str(i)))
        ax[2].legend(handles=handles, bbox_to_anchor=(1.2, 1), borderaxespad=0, title='Cluster ID', labels=range(7), fontsize=8)

    return fig, ax

fig, ax = boxplots_by_cluster_group(adata, ycol='roi_rfp_pos_frac', huecol='kmeans_7', ylabel='ROI RFP+ fraction', title='Cancer cell abundance near SIFT keypoints')
fig.savefig('figures/fig3/rfp_frac_boxplots.pdf', bbox_inches='tight', dpi=300)
plt.show()
fig, ax = boxplots_by_cluster_group(adata, ycol='distance_from_center', huecol='kmeans_7', ylabel='Distance from center (μm)', title='SIFT keypoint position in well')
fig.savefig('figures/fig3/keypoint_position_boxplots.pdf', bbox_inches='tight', dpi=300)
plt.show()
```

```python
# create a boxplot of the ROI RFP+ fraction vs cluster group
fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)

# remove the mixed cluster group from this plot
plot_data = adata.obs[adata.obs['cluster_group'] != 'mixed']
# sort the plot data in the order of cluster_group such hat 'edges' are first, 'singlets' are second, and 'aggregates' are last
plot_data['cluster_group'] = pd.Categorical(plot_data['cluster_group'], categories=['edges', 'singlets', 'aggregates'], ordered=True)
plot_data = plot_data.sort_values('cluster_group')

sns.boxplot(data=plot_data, x='cluster_group', y='roi_rfp_pos_frac', legend=False)

# add the p-value to the plot
ttest = stats.ttest_ind(plot_data.loc[plot_data['cluster_group'] == 'edges','roi_rfp_pos_frac'], plot_data.loc[plot_data['cluster_group'] == 'singlets','roi_rfp_pos_frac'])
print(ttest)
p_adj = ttest.pvalue * 2
ax.text(0.33, 0.99, f'p={p_adj:.1e}', transform=ax.transAxes, ha='center', va='top', fontsize=8)

ttest = stats.ttest_ind(plot_data.loc[plot_data['cluster_group'] == 'singlets','roi_rfp_pos_frac'], plot_data.loc[plot_data['cluster_group'] == 'aggregates','roi_rfp_pos_frac'])
print(ttest)
p_adj = ttest.pvalue * 2
ax.text(0.66, 0.99, f'p={p_adj:.1e}', transform=ax.transAxes, ha='center', va='top', fontsize=8)

ax.set_xlabel('Cluster group')
ax.set_ylabel('ROI RFP+ fraction')
ax.set_title('Cancer cell abundance near SIFT keypoints')
sns.despine(ax=ax)
plt.show()
```

```python
adata.obs.columns
```

```python
cluster_id = 1
x_axis_var = 'time'
hue_var = 'rasa2ko_titration'
cluster_df = adata.obs[adata.obs['kmeans_7'] == cluster_id]
summary_df = []
for (x_value, hue_value), group in cluster_df.groupby([x_axis_var, hue_var]):
    # count the number of keypoints (rows) in this group
    num_keypoints = group.shape[0]
    # compute the mean and standard deviation of the roi_rfp_pos_frac values in this group
    mean_roi_rfp_pos_frac = group['roi_rfp_pos_frac'].mean()
    std_roi_rfp_pos_frac = group['roi_rfp_pos_frac'].std()
    # compute the mean and standard deviation of the roi_rfp_morans_I values in this group
    mean_roi_rfp_morans_I = group['roi_rfp_morans_I'].mean()
    std_roi_rfp_morans_I = group['roi_rfp_morans_I'].std()
    # compute the mean and standard deviation of the roi_bf_morans_I values in this group
    mean_roi_bf_morans_I = group['roi_bf_morans_I'].mean()
    std_roi_bf_morans_I = group['roi_bf_morans_I'].std()
    # save these values to a dataframe
    temp_df = pd.DataFrame({x_axis_var: [x_value],
                            hue_var: [hue_value],
                            'num_keypoints': [num_keypoints],
                            'roi_rfp_pos_frac_mean': [mean_roi_rfp_pos_frac],
                            'roi_rfp_pos_frac_stdev': [std_roi_rfp_pos_frac],
                            'roi_rfp_morans_I_mean': [mean_roi_rfp_morans_I],
                            'roi_rfp_morans_I_stdev': [std_roi_rfp_morans_I],
                            'roi_bf_morans_I_mean': [mean_roi_bf_morans_I],
                            'roi_bf_morans_I_stdev': [std_roi_bf_morans_I]})
    summary_df.append(temp_df)
summary_df = pd.concat(summary_df, ignore_index=True)
summary_df
```

```python
# plot the number of cluster X keypoints over time, split by experimental condition

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def plot_confidence_interval(adata, x_col, y_col, hue_col, cluster_id,
                             ax=None, confidence=0.95, 
                             max_x=np.inf, min_y=-np.inf, max_y=np.inf,
                             title=None, x_label=None, y_label=None,
                             palette='viridis'):
    
    if ax is None:
        fig, ax = plt.subplots()
    
    cluster_df = adata.obs[adata.obs['kmeans_7'] == cluster_id]

    # set the color palette
    pal = sns.color_palette(palette, n_colors=len(cluster_df[hue_col].unique()))
    i = 0
    for hue_value, df in cluster_df.groupby(hue_col):
        # find the unique x values, sort them, and take only the x values that are less than max_x
        unique_xs = df[x_col].unique()
        unique_xs = np.sort(unique_xs)
        unique_xs = unique_xs[unique_xs <= max_x]

        # placeholder for the mean, lower bound, and upper bound
        interval_holder0 = []
        for x in unique_xs:
            # find the mean and the confidence interval for each x value
            temp_df = df[df[x_col] == x]
            mean0, blb0, bub0 = mean_confidence_interval(temp_df[y_col], confidence=confidence)
            interval0 = [mean0, max(blb0, min_y), min(bub0, max_y)]
            interval_holder0.append(interval0)

        # convert the placeholder to an array
        interval_holder0 = np.array(interval_holder0)

        # plot the confidence interval
        ax.fill_between(unique_xs, interval_holder0[:,1], interval_holder0[:,2], alpha=0.6, label=hue_value, color=pal[i])

        i += 1

    sns.despine(ax=ax)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.legend(title=hue_col)


```

```python
clusters_to_plot = [1, 4, 5, 6]

fig, ax = plt.subplots(2, len(clusters_to_plot), figsize=(3*len(clusters_to_plot), 6), tight_layout=True)

for i, cluster_id in enumerate(clusters_to_plot):
    plot_confidence_interval(adata, x_col='time', y_col='roi_rfp_pos_frac', hue_col='rasa2ko_titration', cluster_id=cluster_id,
                             title=f'Cluster {cluster_id}: RFP+ pixel fraction over time',
                             x_label='Time (frame)', y_label='RFP+ pixel fraction (95% CI)',
                             ax=ax[0, i], confidence=0.95, max_x=64, min_y=0, max_y=1, palette='PuBu')
    plot_confidence_interval(adata, x_col='time', y_col='roi_rfp_pos_frac', hue_col='et_ratio', cluster_id=cluster_id,
                             title=f'Cluster {cluster_id}: RFP+ pixel fraction over time',
                             x_label='Time (frame)', y_label='RFP+ pixel fraction (95% CI)',
                             ax=ax[1, i], confidence=0.95, max_x=64, min_y=0, max_y=1, palette='YlGn')

plt.show()
```

```python
def plot_keypoint_count_over_time(adata, cluster_id, x_axis_var, hue_var, 
                                  ax=None, confidence=0.95,
                                  max_x=np.inf,
                                  title=None,
                                  x_label=None,
                                  y_label=None,
                                  palette='viridis'):
    # subset the adata object to just the rows from the specified cluster_id
    cluster_df = adata.obs[adata.obs['kmeans_7'] == cluster_id]

    # create a dataframe to hold the summary statistics
    summary_df = []
    # group the dataframe by the x_axis_var and hue_var
    # and compute the mean and standard deviation of the roi_rfp_pos_frac values in each group
    for (x_value, hue_value), group in cluster_df.groupby([x_axis_var, hue_var]):
        # count the number of keypoints (rows) in this group
        num_keypoints = group.shape[0]
        # save these values to a dataframe
        temp_df = pd.DataFrame({x_axis_var: [x_value],
                                hue_var: [hue_value],
                                'num_keypoints': [num_keypoints]})
        summary_df.append(temp_df)
    summary_df = pd.concat(summary_df, ignore_index=True)

    if ax is None:
        fig, ax = plt.subplots()

    # plot the y_col values over time, split by hue_var
    plot_confidence_interval(summary_df, x_col=x_axis_var, y_col='num_keypoints', hue_col=hue_var, 
                             ax=ax, confidence=confidence, max_x=max_x, palette=palette)

    sns.despine(ax=ax)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.legend(title=hue_var)


# plot_keypoint_count_over_time(adata, cluster_id=1, x_axis_var='time', hue_var='rasa2ko_titration')
```

### Show the `k=7` clustering results alongside the other experimental covariates

This figure will accompany the statistical testing to show which clusters are enriched or depleted in certain experimental conditions. We expect there to be very minimal effect of donor and replicate (i.e. they are the negative controls) but there should be some effects over ratio, titration, and time.

```python
# create a matrix of SIFT PCA embeddings, annotating the points by donor_id, time, rasa2ko_titration, et_ratio
fig, ax = plt.subplots(2, 3, figsize=(8, 4), tight_layout=True)
ax = ax.flatten()

hue_cols = ['kmeans_7', 'replicate_id', 'donor_id', 'time', 'rasa2ko_titration', 'et_ratio']
hue_titles = ['K-means', 'Replicate ID', 'Donor ID', 'Time (hour)', 'RASA2KO\ntitration', 'E:T ratio']
cmaps = ['Dark2', 'Dark2', 'Dark2', 'flare', 'PuBu', 'YlGn']
for i, hue in enumerate(hue_cols):
    # randomize the order of adata.obs so that one color isn't consistently plotted on top of another
    # plot_data = adata.obs.sample(frac=1)
    sns.scatterplot(data=adata.obs, x='PC1', y='PC2', hue=hue, palette=cmaps[i], ax=ax[i], legend=True, alpha=0.7, s=1, rasterized=True)
    ax[i].set_title('SIFT embedding (D={})'.format(num_rows))
    ax[i].set_xlabel('PC1')
    ax[i].set_ylabel('PC2')
    sns.despine(ax=ax[i])
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=hue_titles[i], markerscale=5)

fig.savefig('figures/fig4/sift_pca_embedding_matrix.pdf', bbox_inches='tight', dpi=300)

plt.show()
```

## Statistical testing

Perform a series of statistical tests to determine whether certatin clusters are enriched or depleted for their number of SIFT descriptors (i.e. rows in the dataframe) against a series of covariates `'donor_id', 'time', 'well_id', 'rasa2ko_titration', 'et_ratio'`.

```python
# Perform enrichment analysis of clusters x experimental conditions
results_df = cluster_enrichment(adata.obs, cluster_column='kmeans_7', 
                                continuous_vars=['time', 'et_ratio', 'rasa2ko_titration'], 
                                categorical_vars=['donor_id', 'replicate_id'])
results_df
```

#### Explanation of statistical tests
1. Kruskal-Wallis test is used for continuous variables (`time`, `et_ratio`, `rasa2ko_titration`).
    - Use Kruskal-Wallis (KW) test: Since time may not be normally distributed, KW is a non-parametric alternative to ANOVA that compares medians across clusters.
    - Effect size: Cohen’s d (standardized mean difference).
        - Measures how much the mean time value differs between the cluster and non-cluster points.
        - Positive d = cluster has later times, negative d = cluster has earlier times.
    - `var_value = None` because continuous variables don’t have discrete values like categorical ones.
2. Chi-square or Fisher’s exact test is used for categorical variables (`donor_id`, `replicate_id`).
    - Iterate through each cluster (kmeans_7) and each variable (ratio, titration, etc.).
    - For each unique value of the variable, create a 2×2 contingency table comparing its presence in the cluster vs. outside it.
    - Perform a statistical test:
        - Fisher’s exact test for small sample sizes (when any cell <5).
        - Chi-square test for larger samples.
    - Compute effect size:
        - Odds ratio measures enrichment/depletion.
        - Log2 transformation makes it more interpretable:
            - Positive = enriched in the cluster
            - Negative = depleted in the cluster
3. Benjamini-Hochberg (FDR) correction is applied to all p-values to control false discovery rate.

#### Example interpretation
- Cluster 1 has significantly lower time values (p_adj = 0.005, Cohen’s d = -1.2).
- Cluster 1 is enriched for donor 1 (p_adj = 0.007, log2 odds ratio = 2.5).
- Cluster 2 has significantly higher titration values (p_adj = 0.02).
- Cluster 4 shows no significant association with replicate 0 (p_adj = 0.87).

```python
results_df[results_df['p_adj'] < 0.05].var_name.value_counts()
```

```python
results_df[(results_df['var_name'] == 'time')]
```

```python
results_df[(results_df['var_name'] == 'replicate_id')]
```

```python
results_df[(results_df['var_name'] == 'donor_id')]
```

```python
results_df[(results_df['var_name'] == 'rasa2ko_titration')]
```

```python
results_df[(results_df['var_name'] == 'et_ratio')]
```

```python
import sys
sys.float_info.min
```

```python


# replace NaN values of p_adj with the smallest possible float
results_df['p_adj'] = results_df['p_adj'].fillna(sys.float_info.min)
# replace 0 values of p_adj with the smallest possible float
results_df['p_adj'] = results_df['p_adj'].replace(0, sys.float_info.min)
# recompute the -log10(p_adj) now that NaN values have been replaced
results_df['-log10(p_adj)'] = -np.log10(results_df['p_adj'])

results_df[(results_df['var_name'] == 'et_ratio')]
```

```python
# make a volcano plot of results_df where we show -log10(p_adj) on the y axis
# and the effect_size on the x axis
fig, ax = plt.subplots(1, 3, figsize=(8, 2), tight_layout=True)
ax = ax.flatten()

hue_cols = ['var_name', 'var_value', 'kmeans_7']
hue_titles = ['Variable', 'Value', 'K-means']
cmaps = ['tab10', 'viridis', 'Dark2']
for i, hue_col in enumerate(hue_cols):
    sns.scatterplot(data=results_df, x='effect_size', y='-log10(p_adj)', hue=hue_col, ax=ax[i], s=5, palette=cmaps[i])
    hue_title = hue_titles[i]
    ax[i].set_xlabel('Effect Size')
    ax[i].set_ylabel('-log10(p_adj)')
    sns.despine(ax=ax[i])
    ax[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=hue_title)

plt.show()
```

```python
results_df
```

```python
# an alternative volcano plot is one that colors all points with -log10(p_adj) <10 as grey and then
# labels all the other points with text showing the var_name, var_value, and kmeans_7 values
fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
ax = ax.flatten()

color_threshold = 10
text_threshold = 50
text_threshold = np.inf

var_name_dict = {'time': 'time', 'donor_id': 'donor', 'replicate_id': 'replicate',
                 'rasa2ko_titration': 'RASA2KO', 'et_ratio': 'E:T'}

results_df['covariate'] = results_df['var_name'].map(var_name_dict)

# create a unique colormap of all the unique covariates
covariates = results_df['covariate'].unique()
cmap = plt.cm.get_cmap('Dark2', len(covariates))
colors = cmap(np.arange(len(covariates)))

# cmap = dict(zip(covariates, colors))

cmap = {
    'time': '#9A3671',
    'donor': 'C5',
    'replicate': 'C7',
    'RASA2KO': '#6488A4',
    'E:T': '#7AB85C'
}

# subset results_df to only such that chi2 and kruskal tests are separate dataframes
plot1_df = results_df[results_df['test_type'] == 'chi2']
plot2_df = results_df[results_df['test_type'] == 'kruskal']

for i, plot_df in enumerate([plot1_df, plot2_df]):

    # # draw all the non-significant points in grey
    # sns.scatterplot(data=plot_df[plot_df['-log10(p_adj)'] < color_threshold], x='effect_size', y='-log10(p_adj)', ax=ax[i], color='grey', s=20)
    # # draw all the significant points, annotating each point with its kmeans_7, var_name, and var_value
    # sns.scatterplot(data=plot_df[plot_df['-log10(p_adj)'] >= color_threshold], x='effect_size', y='-log10(p_adj)', ax=ax[i], hue='covariate', palette=cmap, s=20)

    # draw all points with the color annotating the covariate
    sns.scatterplot(data=plot_df, x='effect_size', y='-log10(p_adj)', ax=ax[i], hue='covariate', palette=cmap, s=20)

    for _, row in plot_df[plot_df['-log10(p_adj)'] >= text_threshold].iterrows():
        if row['covariate'] == 'time':
            ax[i].annotate('cluster={}\nlate times'.format(row['kmeans_7']), (row['effect_size'], row['-log10(p_adj)']),
                        ha='center', va='center', fontsize=SMALL_SIZE, color='black', xytext=(0, 10), textcoords='offset points')
        elif row['covariate'] == 'E:T':
            ax[i].annotate('cluster={}\nhigh E:T'.format(row['kmeans_7']), (row['effect_size'], row['-log10(p_adj)']),
                        ha='center', va='center', fontsize=SMALL_SIZE, color='black', xytext=(0, 10), textcoords='offset points')
        elif row['covariate'] == 'RASA2KO':
            ax[i].annotate('cluster={}\nhigh RASA2KO'.format(row['kmeans_7']), (row['effect_size'], row['-log10(p_adj)']),
                        ha='center', va='center', fontsize=SMALL_SIZE, color='black', xytext=(0, 10), textcoords='offset points')
        else:
            ax[i].annotate('cluster={}\n{}={}'.format(row['kmeans_7'], row['covariate'], round(row['var_value'], 2)), (row['effect_size'], row['-log10(p_adj)']),
                        ha='center', va='center', fontsize=SMALL_SIZE, color='black', xytext=(0, 10), textcoords='offset points')
    
    # draw a dashed horizontal line at y=50
    ax[i].hlines(y=50, xmin=-0.6, xmax=0.6, colors='grey', linestyles='dashed', alpha=0.5)

    sns.despine(ax=ax[i])
    ax[i].legend(title='coviariate', loc='lower left', markerscale=1)
    ax[i].set_ylim(-10, 250)


ax[0].set_xlabel('Effect size (log$_2$ odds ratio)\n<--depleted | enriched-->')
ax[0].set_ylabel('Chi-square test\n-log$\mathrm{_{10}(p_{adj})}$')
ax[0].set_title('Categorical covariates')

ax[1].set_xlabel("Effect size (Cohen's D)\n<--depleted | enriched-->")
ax[1].set_ylabel('Kruskal-Wallis test\n-log$\mathrm{_{10}(p_{adj})}$')
ax[1].set_title('Continuous covariates')

fig.savefig('figures/fig4/volcanos.pdf', bbox_inches='tight', dpi=300)

plt.show()
```

```python
# save results_df to analysis file
results_df.to_csv('analysis/sift_volcano_table.csv')
```

### Look into relationship with spatial entropy and number of aggregate keypoints (clusters 5 + 6) in each image

```python
adata_full.obs.columns.values
```

```python
adata.obs.columns.values
```

```python
adata_full.obs.shape
```

```python
adata.obs[['donor_id', 'time', 'well_id', 'rasa2ko_titration', 'et_ratio', 'entropy', 'p_areas']].drop_duplicates().shape
```

```python
adata.obs.filename.unique().shape
```

```python
image_df = []

# loop through each unique image (filename)
for filename, chunk in adata_full.obs.groupby('filename'):

    # create a temporary dataframe to hold the donor_id, time, well_id, rasa2ko_titration, et_ratio, entropy, p_areas for this image
    temp_df = pd.DataFrame(chunk[['donor_id', 'time', 'well_id', 'rasa2ko_titration', 'et_ratio', 'entropy', 'p_areas', 'filename']].drop_duplicates())

    # count the number of keypoints belonging to each kmeans_7 cluster
    cluster_counts = chunk.groupby('kmeans_7').size()
    # add the counts to the temporary dataframe
    for i, count in cluster_counts.items():
        temp_df['n_keypoints_cluster_{}'.format(i)] = count

    # append the temporary dataframe to the image_df
    image_df.append(temp_df)

image_df = pd.concat(image_df, ignore_index=True)
image_df
```

```python
image_df['n_keypoints_aggregates'] = image_df['n_keypoints_cluster_5'] + image_df['n_keypoints_cluster_6']
image_df['n_keypoints_singlets'] = image_df['n_keypoints_cluster_1'] + image_df['n_keypoints_cluster_4']
image_df['n_keypoints_edges'] = image_df['n_keypoints_cluster_2'] + image_df['n_keypoints_cluster_3']
image_df['n_keypoints_total'] = image_df['n_keypoints_aggregates'] + image_df['n_keypoints_singlets'] + image_df['n_keypoints_edges'] + image_df['n_keypoints_cluster_0']

# compute the fraction of keypoints that are singlets, aggregates, and edges
image_df['frac_singlets'] = image_df['n_keypoints_singlets'] / image_df['n_keypoints_total']
image_df['frac_aggregates'] = image_df['n_keypoints_aggregates'] / image_df['n_keypoints_total']
image_df['frac_edges'] = image_df['n_keypoints_edges'] / image_df['n_keypoints_total']

image_df
```

```python
fig, ax = plt.subplots(2, 3, figsize=(8, 6), tight_layout=True)
ax = ax.ravel()

def plot_scatter_with_line_of_best_fit(ax, x, y, hue, alpha=0.5, s=1):
    sns.scatterplot(ax=ax, x=x, y=y, hue=hue, alpha=alpha, s=s)
    ax = sns.regplot(ax=ax, x=x, y=y, scatter=False, line_kws={'color': 'grey', 'lw': 1, 'ls': '--'})
    result = stats.pearsonr(x, y)
    ax.text(0.5, 0.9, 'r={:.2f}, p={:.2e}'.format(result[0], result[1]), transform=ax.transAxes, fontsize=8, ha='center', va='center')
    return ax

plot_scatter_with_line_of_best_fit(ax[0], image_df['frac_singlets'], image_df['entropy'], image_df['et_ratio'], alpha=0.5, s=1)
plot_scatter_with_line_of_best_fit(ax[1], image_df['frac_aggregates'], image_df['entropy'], image_df['et_ratio'], alpha=0.5, s=1)
plot_scatter_with_line_of_best_fit(ax[2], image_df['frac_edges'], image_df['entropy'], image_df['et_ratio'], alpha=0.5, s=1)

plot_scatter_with_line_of_best_fit(ax[3], image_df['frac_singlets'], image_df['p_areas'], image_df['et_ratio'], alpha=0.5, s=1)
plot_scatter_with_line_of_best_fit(ax[4], image_df['frac_aggregates'], image_df['p_areas'], image_df['et_ratio'], alpha=0.5, s=1)
plot_scatter_with_line_of_best_fit(ax[5], image_df['frac_edges'], image_df['p_areas'], image_df['et_ratio'], alpha=0.5, s=1)

# # plot the fraction of the 3 cluster categories vs entropy
# sns.scatterplot(ax=ax[0],data=image_df, x='frac_singlets', y='entropy', hue='et_ratio', alpha=0.5, s=1)
# sns.scatterplot(ax=ax[1], data=image_df, x='frac_aggregates', y='entropy', hue='et_ratio', alpha=0.5, s=1)
# sns.scatterplot(ax=ax[2],data=image_df, x='frac_edges', y='entropy', hue='et_ratio', alpha=0.5, s=1)

# # plot the fraction of the 3 cluster categories vs p_areas
# sns.scatterplot(ax=ax[3],data=image_df, x='frac_singlets', y='p_areas', hue='et_ratio', alpha=0.5, s=1)
# sns.scatterplot(ax=ax[4], data=image_df, x='frac_aggregates', y='p_areas', hue='et_ratio', alpha=0.5, s=1)
# sns.scatterplot(ax=ax[5],data=image_df, x='frac_edges', y='p_areas', hue='et_ratio', alpha=0.5, s=1)



for a in ax:
    sns.despine(ax=a)
    a.set_title('N={} images'.format(image_df.shape[0]))
    a.legend(title='E:T ratio', markerscale=5)

for i in range(0, 3):
    ax[i].set_ylabel('RFP spatial entropy')
for i in range(3, 6):
    ax[i].set_ylabel('RFP+ area (μm$^2$)')
for i in [0, 3]:
    ax[i].set_xlabel('Fraction of SIFT keypoints that are singlets')
for i in [1, 4]:
    ax[i].set_xlabel('Fraction of SIFT keypoints that are aggregates')
for i in [2, 5]:
    ax[i].set_xlabel('Fraction of SIFT keypoints that are edges')

plt.show()
```

```python

```
