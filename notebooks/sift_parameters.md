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
from sklearn.metrics import silhouette_score
import scipy.stats as stats
from sflcba.stats import cluster_enrichment
from skimage.feature import SIFT

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
adata_all = ad.read_h5ad('analysis/adata_20250225_processed_20250310.h5ad')

# the number of rows when first loading the data represents the number of SIFT descriptors used for clustering
num_rows_cluster = adata_all.obs.shape[0]

adata_all
```

```python
# image spatialresolution is 4.975 um per pixel
um_per_pixel = 4.975

# convert the x and y coordinates from pixels to um
adata_all.obs['x_um'] = adata_all.obs['x'] * um_per_pixel
adata_all.obs['y_um'] = adata_all.obs['y'] * um_per_pixel

# convert the p_area from pixels^2 to μm$^2$
adata_all.obs['p_areas'] = adata_all.obs['p_areas'] * (um_per_pixel**2)

# image time resolution is 2 hours per frame
hours_per_frame = 2

# convert the time from frames to hours
adata_all.obs['time'] = adata_all.obs['time'] * hours_per_frame
```

```python
def get_entire_well_image(df, well_id, donor_id, time_point, trim=100, return_filenames=False):
    '''
    Retrieve the entire brightfield and RFP image for a single well.

    Parameters
    ----------
     df : pd.DataFrame
        dataframe of SIFT keypoints and descriptors. Columns should include ['well_id', 'donor_id', 'time', 'filename']
    well_id : str
        well id of the image we wish to plot
    donor_id : str
        donor id of the image we wish to plot
    time_point : int
        time point of the image we wish to plot
    trim : int
        number of pixels to trim from the edges of the image
    return_filenames : bool
        if True, return the filenames of the phase and red images along with the images
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

    if trim > 0:
        # trim the edges of the image
        phase_frame = phase_frame[trim:-trim,trim:-trim]
        red_frame = red_frame[trim:-trim,trim:-trim]
    
    if return_filenames:
        return phase_frame, red_frame, phase_filename, red_filename
    else:
        return phase_frame, red_frame


def plot_entire_well(ax, phase_frame, red_frame=None):
    # plot the phase image with the red mask superimposed
    if red_frame is not None:
        ax.imshow(red_frame, cmap='Reds', alpha = 1.0)
    ax.imshow(phase_frame, cmap='gray', alpha = .75)
    # remove the axes ticks and labels
    ax.axis('off')


fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
phase, red, = get_entire_well_image(adata_all.obs, well_id='B4', donor_id=1, time_point=90, trim=100, return_filenames=False)
plot_entire_well(ax, phase, red)
ax.set_title('Donor 1, Well B4, Time 90h')
plt.show()
```

```python
help(SIFT)
```

```python
def compute_sift_embedding(phase_image, downsample_pct=1.0, sift=SIFT()):
    """
    Given the path to a phase image file, compute the SIFT keypoints and descriptors.
    This function loads the image, runs the SIFT detector, and downsamples the results.
    Returns a dictionary with the embedding and associated metadata.
    """
    # Load image and normalize intensity
    phase_frame = cv.normalize(phase_image, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    
    # Detect SIFT keypoints and compute descriptors
    sift.detect_and_extract(phase_frame)
    keypoints = sift.keypoints
    descriptors = sift.descriptors
    scales = sift.scales
    octaves = sift.octaves
    sigmas = sift.sigmas
    orientations = sift.orientations
    
    # Downsample descriptors to reduce memory footprint
    n_keypoints = len(keypoints)
    idx = np.random.choice(n_keypoints, int(n_keypoints * downsample_pct), replace=False)
    
    # Package results in a dictionary (you can also construct an AnnData object here)
    embed = {
        "x": keypoints[idx, 0],
        "y": keypoints[idx, 1],
        "descriptors": descriptors[idx],
        "scales": scales[idx],
        "octaves": octaves[idx],
        "sigmas": sigmas[idx],
        "orientations": orientations[idx],
        "n_keypoints_original": n_keypoints
    }

    adata = ad.AnnData(X=embed["descriptors"])

    # Attach metadata to the AnnData object.
    adata.obs['scales'] = embed["scales"]
    adata.obs['octaves'] = embed["octaves"]
    adata.obs['sigmas'] = embed["sigmas"]
    adata.obs['orientations'] = embed["orientations"]
    adata.obs['x'] = embed["x"]
    adata.obs['y'] = embed["y"]
    adata.obs['n_og_keypoints'] = embed["n_keypoints_original"]
    adata.obs['roi_radius'] = adata.obs['scales'] * (2 ** (adata.obs['octaves'] + 1))

    # add a sift_ prefix to the variable names
    adata.var_names = ['sift_{}'.format(i) for i in range(128)]

    return adata


adata_default = compute_sift_embedding(phase, sift=SIFT())
adata_default
```

```python
def plot_large_roi_w_keypoints(df, ax, roi_center, roi_radius, phase_image, rfp_image=None, um_per_pixel=4.975, color='b'):
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

    # find the x_min, x_max, y_min, y_max boundaries of the image based on the roi_center and roi_radius
    x_min = roi_center[0] - roi_radius
    x_max = roi_center[0] + roi_radius
    y_min = roi_center[1] - roi_radius
    y_max = roi_center[1] + roi_radius

    ax.imshow(rfp_image[x_min:x_max, y_min:y_max], cmap='Reds', alpha = 1.0)
    ax.imshow(phase_image[x_min:x_max, y_min:y_max], cmap='gray', alpha = .75)

    # subset the dataframe to only include keypoints within the ROI
    df_roi = df[(df['x'] >= x_min) & (df['x'] <= x_max) & (df['y'] >= y_min) & (df['y'] <= y_max)]

    color_is_column = color in df.columns

    # find all the SIFT keypoints in this image (each row in df) and annotate their location
    for idx, row in df_roi.iterrows():
        # check to see if the x-y position of this SIFT keypoint falls within xx_min:x_max, y_min:y_max
        key_x = row['x']
        key_y = row['y']
        # if (key_x < x_max) and (key_x > x_min) and (key_y < y_max) and (key_y > y_min):
        key_radius = row['roi_radius']
        if color_is_column:
            # set color of title and circle based on the cluster_col value and the Dark2 colormap
            edgecolor = cm.Dark2(row[color])
        else:
            edgecolor = color
        # annotate the ROI with a circle that has r=radius
        circle = patches.Circle((key_x-x_min, key_y-y_min), key_radius, edgecolor=edgecolor, facecolor='none')
        ax.add_patch(circle)
    
        # add a line going from the center of the circle to the perimeter of the circle and an angle of row['orientations']
        angle = row['orientations']
        ax.plot([key_x-x_min, key_x-x_min + key_radius * np.cos(angle)], [key_y-y_min, key_y-y_min + key_radius * np.sin(angle)], color=edgecolor)

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
    
    return ax, df_roi

fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
ax, df_roi_default = plot_large_roi_w_keypoints(adata_default.obs, ax, roi_center=(300,300), roi_radius=200, phase_image=phase, rfp_image=red)
ax.set_title(f'Default SIFT parameters\nD={df_roi_default.shape[0]} keypoints in ROI')
plt.show()
```

```python
phase.shape
```

```python
# compute SIFT embeddings with upsampling set to 1 and 4
adata_up1 = compute_sift_embedding(phase, sift=SIFT(upsampling=1, sigma_in=0.0))
adata_up2 = compute_sift_embedding(phase, sift=SIFT(upsampling=2, sigma_in=0.0))
adata_up4 = compute_sift_embedding(phase, sift=SIFT(upsampling=4, sigma_in=0.0))

# plot the keypoints for the upsampling=1, upsampling=2 (default), and upsampling=4 superimposed on the same ROI
fig, ax = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)
ax[0], df_roi_up1 = plot_large_roi_w_keypoints(adata_up1.obs, ax[0], roi_center=(500,500), roi_radius=200, phase_image=phase, rfp_image=red)
ax[1], df_roi_up2 = plot_large_roi_w_keypoints(adata_up2.obs, ax[1], roi_center=(500,500), roi_radius=200, phase_image=phase, rfp_image=red)
ax[2], df_roi_up4 = plot_large_roi_w_keypoints(adata_up4.obs, ax[2], roi_center=(500,500), roi_radius=200, phase_image=phase, rfp_image=red)
ax[0].set_title('upsampling=1, sigma_in=0\nD={} keypoints in ROI'.format(df_roi_up1.shape[0]))
ax[1].set_title('upsampling=2 (default), sigma_in=0\nD={} keypoints in ROI'.format(df_roi_up2.shape[0]))
ax[2].set_title('upsampling=4, sigma_in=0\nD={} keypoints in ROI'.format(df_roi_up4.shape[0]))
plt.show()
```

Do an experiment where I crop the image prior to passing it into SIFT to see if any of the detected keypoints change.

```python
roi_center=(500,500)
roi_radius=200

cropped_phase = phase[roi_center[0]-roi_radius:roi_center[0]+roi_radius, roi_center[1]-roi_radius:roi_center[1]+roi_radius]
cropped_red = red[roi_center[0]-roi_radius:roi_center[0]+roi_radius, roi_center[1]-roi_radius:roi_center[1]+roi_radius]

# compute SIFT embeddings on the cropped image
adata_cropped = compute_sift_embedding(cropped_phase, sift=SIFT())

# plot the keypoints for the cropped image superimposed on the same ROI
fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)

ax[0], df_roi_default = plot_large_roi_w_keypoints(adata_default.obs, ax[0], roi_center=roi_center, roi_radius=roi_radius, phase_image=phase, rfp_image=red)
ax[0].set_title(f'Full image passed into SIFT\nDefault SIFT parameters\nD={df_roi_default.shape[0]} keypoints in ROI')

# this ROI center is based on the already cropped image so the center should always be (roi_radius,roi_radius)
ax[1], df_roi_cropped = plot_large_roi_w_keypoints(adata_cropped.obs, ax[1], roi_center=(roi_radius,roi_radius), roi_radius=roi_radius, phase_image=cropped_phase, rfp_image=cropped_red)
ax[1].set_title(f'Cropped ROI passed into SIFT\nDefault SIFT parameters\nD={df_roi_cropped.shape[0]} keypoints in ROI')

# copy the ax[0] tick labels for ax[1]
ax[1].set_xticklabels(ax[0].get_xticklabels())
ax[1].set_yticklabels(ax[0].get_yticklabels())

fig.savefig('figures/fig3/SIFT_cropped_vs_full_image.pdf', dpi=300, bbox_inches='tight')

plt.show()

```

```python
df_roi_cropped
```

```python
df_roi_default.x.describe()
```

```python
df_roi_cropped.x.describe()
```

```python
df_roi_cropped.y.describe()
```

```python
df_roi_default.y.describe()
```

```python
df_roi_default
```

```python
# align the coordinates of df_roi_cropped to match those of df_roi_default
df_roi_cropped['x'] = df_roi_cropped['x'] + 300
df_roi_cropped['y'] = df_roi_cropped['y'] + 300

# round orientations and sigmas to 1 decimal place before merging
df_roi_default['orientations'] = df_roi_default['orientations'].round(1)
df_roi_cropped['orientations'] = df_roi_cropped['orientations'].round(1)

df_roi_default['sigmas'] = df_roi_default['sigmas'].round(1)
df_roi_cropped['sigmas'] = df_roi_cropped['sigmas'].round(1)
```

```python
pd.merge(df_roi_default, df_roi_cropped, how='inner', on=['octaves', 'scales', 'x', 'y', 'orientations'], suffixes=('_default', '_cropped'))
```

```python
# create a venn diagram showing the overlap between the two sets of keypoints
# start by computing the number of keypoints in each set and the intersection of the two sets
n_keypoints_default = len(df_roi_default)
n_keypoints_cropped = len(df_roi_cropped)
n_keypoints_intersection = len(pd.merge(df_roi_default, df_roi_cropped, how='inner', on=['octaves', 'scales', 'x', 'y', 'orientations'], suffixes=('_default', '_cropped')))

print(f'Number of keypoints exclusive to default set: {n_keypoints_default - n_keypoints_intersection}')
print(f'Number of keypoints exclusive to cropped set: {n_keypoints_cropped - n_keypoints_intersection}')
print(f'Number of keypoints in intersection: {n_keypoints_intersection}')

```

```python
# find the intersection of the two dataframes based on all columns
# shift the x and y coordinates of the cropped dataframe by the roi_center values so they match the full image coordinates
temp_cropped_df = adata_cropped.obs.copy()
temp_cropped_df['x'] = temp_cropped_df['x'] + roi_center[0]
temp_cropped_df['y'] = temp_cropped_df['y'] + roi_center[1]
df_intersection = pd.merge(adata_default.obs, temp_cropped_df, how='inner', suffixes=('_default', '_cropped'), on=['scales', 'octaves', 'sigmas'])
df_intersection
```

```python
df_roi_cropped
```

```python
df_roi_default
```

```python

```
