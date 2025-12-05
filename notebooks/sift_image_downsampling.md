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

This notebook is dedicated to benchmarking how SIFT performs when images are downsampled to even lower resolution or the SIFT parameters are changed.

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import SIFT
from sflcba.sift import add_pca_embedding
from sklearn.cluster import KMeans


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
# adata = ad.read_h5ad('analysis/adata_processed.h5ad')
adata = ad.read_h5ad('analysis/adata_20250225_processed_20250310.h5ad')

# image spatial resolution is 4.975 um per pixel
um_per_pixel = 4.975
# convert the x and y coordinates from pixels to um
adata.obs['x_um'] = adata.obs['x'] * um_per_pixel
adata.obs['y_um'] = adata.obs['y'] * um_per_pixel
# convert the p_area from pixels^2 to μm$^2$
adata.obs['p_areas'] = adata.obs['p_areas'] * (um_per_pixel**2)

# image time resolution is 2 hours per frame
hours_per_frame = 2
# convert the time from frames to hours
adata.obs['time'] = adata.obs['time'] * hours_per_frame
```

```python
def get_image_roi(df, well_id, donor_id, time_point, roi_center, roi_radius):
    '''
    Given a specified well_id, donor_id, time point, and ROI center and radius, extract the red and phase channels from this ROI.

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
    roi_center : tuple of int
        center of the ROI in x-y coordinates
    roi_radius : int
        radius of the ROI in pixels
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

    red_roi = red_frame[x_min:x_max, y_min:y_max]
    phase_roi = phase_frame[x_min:x_max, y_min:y_max]

    return red_roi, phase_roi

```

```python

red_roi, phase_roi = get_image_roi(adata.obs, well_id='H9', donor_id=1, time_point=30, roi_center=(500,500), roi_radius=200)
fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.imshow(red_roi, cmap='Reds', alpha = 1.0)
ax.imshow(phase_roi, cmap='gray', alpha = .75)
plt.show()
```

```python
def downsample_image(image, factor):
    """Downsample an image by a (possibly non-integer) factor using cv2.resize.

    Parameters
    ----------
    image : np.ndarray
        input image to be downsampled (2D or 3D)
    factor : float
        downsampling factor (>1 reduces size, <1 enlarges)

    Returns
    -------
    np.ndarray
        resulting image
    """
    if factor <= 0:
        raise ValueError("factor must be > 0")

    height, width = image.shape[:2]
    new_height = max(1, int(np.round(height / factor)))
    new_width = max(1, int(np.round(width / factor)))

    # choose interpolation: INTER_AREA for shrinking, INTER_LINEAR for enlarging
    interp = cv.INTER_AREA if factor >= 1 else cv.INTER_LINEAR

    downsampled_image = cv.resize(image, (new_width, new_height), interpolation=interp)
    return downsampled_image

downsampled_1_5x_phase = downsample_image(phase_roi, factor=1.5)
downsampled2x_phase = downsample_image(phase_roi, factor=2)
downsampled4x_phase = downsample_image(phase_roi, factor=4)
fig, ax = plt.subplots(1, 4, figsize=(16,4))
ax = ax.ravel()
ax[0].imshow(phase_roi, cmap='gray')
ax[0].set_title('Original resolution ({um_per_pixel:.3f} um/pixel)'.format(um_per_pixel=um_per_pixel))
ax[1].imshow(downsampled_1_5x_phase, cmap='gray')
ax[1].set_title('1.5x downsampled ({um_per_pixel:.3f} um/pixel)'.format(um_per_pixel=um_per_pixel*1.5))
ax[2].imshow(downsampled2x_phase, cmap='gray')
ax[2].set_title('2x downsampled ({um_per_pixel:.3f} um/pixel)'.format(um_per_pixel=um_per_pixel*2))
ax[3].imshow(downsampled4x_phase, cmap='gray')
ax[3].set_title('4x downsampled ({um_per_pixel:.3f} um/pixel)'.format(um_per_pixel=um_per_pixel*4))
plt.show()
```

```python
def compute_sift_embedding(phase_frame, downsample_pct=1, n_octaves=8, n_scales=3):
    """
    Given the path to a phase image file, compute the SIFT keypoints and descriptors.
    This function loads the image, runs the SIFT detector, and downsamples the results.
    Returns a dictionary with the embedding and associated metadata.
    """
    phase_frame = cv.normalize(phase_frame, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    
    # Detect SIFT keypoints and compute descriptors
    descriptor_extractor = SIFT(n_octaves=n_octaves, n_scales=n_scales)
    try:
        descriptor_extractor.detect_and_extract(phase_frame)
    except:
        # return empty dict if SIFT fails for this image
        return {}
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors
    scales = descriptor_extractor.scales
    octaves = descriptor_extractor.octaves
    sigmas = descriptor_extractor.sigmas
    orientations = descriptor_extractor.orientations
    
    # Downsample descriptors to reduce memory footprint
    n_keypoints = len(keypoints)
    idx = np.random.choice(n_keypoints, int(n_keypoints * downsample_pct), replace=False)
    
    # Package results in a dictionary (you can also construct an AnnData object here)
    result = {
        "x": keypoints[idx, 0],
        "y": keypoints[idx, 1],
        "descriptors": descriptors[idx],
        "scales": scales[idx],
        "octaves": octaves[idx],
        "sigmas": sigmas[idx],
        "orientations": orientations[idx],
        "n_keypoints_original": n_keypoints
    }
    return result

def make_sift_adata(phase_image, image_downsample_factor=1, cluster=True, n_octaves=8, n_scales=3):
    # compute the sift embedding for the phase image
    embed = compute_sift_embedding(phase_image, downsample_pct=1, n_octaves=n_octaves, n_scales=n_scales)
    # Return an empty AnnData object if no descriptors were found.
    if embed == {} or embed["descriptors"].size == 0:
        return ad.AnnData()
    adata = ad.AnnData(X=embed["descriptors"])

    # Attach metadata to the AnnData object.
    adata.obs['image_downsample_factor'] = image_downsample_factor
    adata.obs['scales'] = embed["scales"]
    adata.obs['octaves'] = embed["octaves"]
    adata.obs['sigmas'] = embed["sigmas"]
    adata.obs['orientations'] = embed["orientations"]
    adata.obs['x'] = embed["x"]
    adata.obs['y'] = embed["y"]
    adata.obs['n_og_keypoints'] = embed["n_keypoints_original"]

    # add a sift_ prefix to the variable names
    adata.var_names = ['sift_{}'.format(i) for i in range(128)]

    # # add PCA embedding of the sift descriptors
    # adata = add_pca_embedding(adata, n_components=2)

    # perform K-means clustering of the sift descriptors
    if cluster:
        k = 5
        kmeans = KMeans(n_clusters=k, random_state=0)
        adata.obs['kmeans_' + str(k)] = kmeans.fit_predict(adata.X)

    # compute the ROI radius for each SIFT descriptor
    adata.obs['roi_radius'] = adata.obs['scales'] * (2 ** (adata.obs['octaves'] + 1))

    return adata

# compute the SIFT embedding for the original, 2x downsampled, and 4x downsampled images
adata_original = make_sift_adata(phase_roi, image_downsample_factor=1)
adata_1_5x = make_sift_adata(downsampled_1_5x_phase, image_downsample_factor=1.5)
adata_2x = make_sift_adata(downsampled2x_phase, image_downsample_factor=2)
adata_4x = make_sift_adata(downsampled4x_phase, image_downsample_factor=4)
print('Original image SIFT descriptors:', adata_original.n_obs)
print('1.5x downsampled image SIFT descriptors:', adata_1_5x.n_obs)
print('2x downsampled image SIFT descriptors:', adata_2x.n_obs)
print('4x downsampled image SIFT descriptors:', adata_4x.n_obs)
```

```python
adata_2x.obs.kmeans_5.value_counts()
```

```python
def plot_sift_embedding_with_keypoints(ax, adata, phase_roi, cluster_col='kmeans_5'):
    ax.imshow(phase_roi, cmap='gray', alpha = .75)

    # find all the SIFT keypoints in this image (each row in df) and annotate their location
    for idx, row in adata.obs.iterrows():
        key_x = row['x']
        key_y = row['y']
        key_radius = row['roi_radius']
        # set color of title and circle based on the cluster_col value
        if cluster_col is not None:
            color = 'C{}'.format(int(row[cluster_col]))
        else:
            color = 'C1'
        # annotate the ROI with a circle that has r=radius
        circle = patches.Circle((key_x, key_y), key_radius, edgecolor=color, facecolor='none')
        ax.add_patch(circle)
    
        # add a line going from the center of the circle to the perimeter of the circle and an angle of row['orientations']
        angle = row['orientations']
        ax.plot([key_x, key_x + key_radius * np.cos(angle)], [key_y, key_y + key_radius * np.sin(angle)], color=color)


fig, ax = plt.subplots(1, 4, figsize=(8,2), tight_layout=True)
ax = ax.ravel()
plot_sift_embedding_with_keypoints(ax[0], adata_original, phase_roi, cluster_col='kmeans_5')
ax[0].set_title(f'Original ({um_per_pixel} um/pixel)\nD={adata_original.n_obs} keypoints')
plot_sift_embedding_with_keypoints(ax[1], adata_1_5x, downsampled_1_5x_phase, cluster_col='kmeans_5')
ax[1].set_title(f'1.5x ({um_per_pixel*1.5:.3f} um/pixel)\nD={adata_1_5x.n_obs} keypoints')
plot_sift_embedding_with_keypoints(ax[2], adata_2x, downsampled2x_phase, cluster_col='kmeans_5')
ax[2].set_title(f'2x ({um_per_pixel*2} um/pixel)\nD={adata_2x.n_obs} keypoints')
plot_sift_embedding_with_keypoints(ax[3], adata_4x, downsampled4x_phase, cluster_col='kmeans_5')
ax[3].set_title(f'4x ({um_per_pixel*4} um/pixel)\nD={adata_4x.n_obs} keypoints')
fig.savefig('figures/supp_figs/image_downsampling_roi.pdf', bbox_inches='tight', dpi=300)
plt.show()
```

```python
adata.obs.well_id.unique()
```

```python
adata.obs.time.unique()
```

```python
adata.obs.loc[(adata.obs.well_id == 'H10') & (adata.obs.time == 49)]
```

```python
# downsample 5 different random ROIs at rates of 1x, 1.5x, 2x, 3x, and 4x and compute the number of SIFT descriptors found in each ROI
downsample_factors = [1, 1.25, 1.5, 1.75, 2, 3, 4]
roi_centers = [(random.randint(400, 800), random.randint(400, 800)) for _ in range(5)]
time_points = random.choices(adata.obs.time.unique(), k=5)
# pick five random wells 
wells = random.choices(adata.obs.well_id.unique(), k=5)
roi_radius = 200

summary_df = []

for roi_center, time_point, well in zip(roi_centers, time_points, wells):
    print('center: ', roi_center, 'time point: ', time_point, 'well: ', well)
    red_roi, phase_roi = get_image_roi(adata.obs, well_id=well, donor_id=1, time_point=time_point, roi_center=roi_center, roi_radius=roi_radius)
    # plot the ROI
    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    ax.imshow(red_roi, cmap='Reds', alpha = 1.0)
    ax.imshow(phase_roi, cmap='gray', alpha = .75)
    plt.show()
    for factor in downsample_factors:
        if factor == 1:
            downsampled_phase = phase_roi
        else:
            downsampled_phase = downsample_image(phase_roi, factor=factor)
        adata_sift = make_sift_adata(downsampled_phase, image_downsample_factor=factor, cluster=False)
        summary_df.append({
            'roi_center': roi_center,
            'time_point': time_point,
            'well': well,
            'downsample_factor': factor,
            'um_per_pixel': um_per_pixel * factor,
            'n_sift_descriptors': adata_sift.n_obs
        })

summary_df = pd.DataFrame(summary_df)
summary_df
```

```python
# make a line plot that shows the number of SIFT descriptors vs downsample factor for each ROI
fig, ax = plt.subplots(1, 2, figsize=(8,3), tight_layout=True)
for roi_center, group in summary_df.groupby('roi_center'):
    well = group['well'].iloc[0]
    hour = group['time_point'].iloc[0]
    ax[0].plot(group['um_per_pixel'], group['n_sift_descriptors'], marker='o', label=f'Well: {well}, Time: {hour}, Center: {roi_center}')
    # normalize number of sift descriptors to the number found at 1x downsampling
    normalized_n_sift = group['n_sift_descriptors'] / group[group['downsample_factor'] == 1]['n_sift_descriptors'].values[0]
    ax[1].plot(group['um_per_pixel'], normalized_n_sift, marker='o', label=f'Well: {well}, Time: {hour}, Center: {roi_center}')
ax[0].set_xlabel('um per pixel')
ax[0].set_ylabel('Number of keypoints detected')
# ax[0].legend(title='ROIs')

ax[1].set_xlabel('um per pixel')
ax[1].set_ylabel('Proportion of keypoints relative to original image')
ax[1].legend(title='ROIs', bbox_to_anchor=(1.05, 1), loc='upper left')

fig.savefig('figures/supp_figs/image_downsampling_results.pdf', bbox_inches='tight', dpi=300)
```

Perform a grid search on SIFT parameters `n_octaves` and `n_scales` to assess how they affect the number of SIFT descriptors found in the ROI

```python
# test differente octaves and scales
n_octaves = [3, 8, 13]
n_scales = [1, 3, 5]

roi_centers = [(random.randint(400, 800), random.randint(400, 800)) for _ in range(5)]
time_points = random.choices(adata.obs.time.unique(), k=5)
# pick five random wells 
wells = random.choices(adata.obs.well_id.unique(), k=5)
roi_radius = 200

summary_df = []

counter = 0
for roi_center, time_point, well in zip(roi_centers, time_points, wells):
    print('center: ', roi_center, 'time point: ', time_point, 'well: ', well)
    red_roi, phase_roi = get_image_roi(adata.obs, well_id=well, donor_id=1, time_point=time_point, roi_center=roi_center, roi_radius=roi_radius)

    fig, ax = plt.subplots(3, 3, figsize=(9,9), tight_layout=True)
    for o, octave in enumerate(n_octaves):
        for s, scale in enumerate(n_scales):
            if factor == 1:
                downsampled_phase = phase_roi
            else:
                downsampled_phase = downsample_image(phase_roi, factor=factor)
            adata_sift = make_sift_adata(phase_roi, image_downsample_factor=1, cluster=True, n_octaves=octave, n_scales=scale)

            # compute the min, max, and mean roi radius of the sift descriptors
            if adata_sift.n_obs == 0:
                min_radius = np.nan
                max_radius = np.nan
                mean_radius = np.nan
            else:
                min_radius = adata_sift.obs['roi_radius'].min()
                max_radius = adata_sift.obs['roi_radius'].max()
                mean_radius = adata_sift.obs['roi_radius'].mean()

            plot_sift_embedding_with_keypoints(ax[s,o], adata_sift, phase_roi, cluster_col='kmeans_5')
            ax[s,o].set_title(f'Octaves: {octave}, Scales: {scale}\nD={adata_sift.n_obs}')
            ax[s,o].axis('off')

            summary_df.append({
                'roi_center': roi_center,
                'time_point': time_point,
                'well': well,
                'n_octaves': octave,
                'n_scales': scale,
                'min_roi_radius': min_radius,
                'max_roi_radius': max_radius,
                'mean_roi_radius': mean_radius,
                'n_sift_descriptors': adata_sift.n_obs
            })
    
    fig.savefig('figures/supp_figs/sift_gridsearch_roi_{}.pdf'.format(counter), bbox_inches='tight', dpi=300)
    plt.show()
    counter += 1

summary_df = pd.DataFrame(summary_df)
summary_df
```

```python
# look at the meam roi radius and number of sift descriptors for each combination of n_octaves and n_scales in the gridsearch
# plot this as a heatmap for each ROI x target variable
for roi_center, group in summary_df.groupby('roi_center'):
    well = group['well'].iloc[0]
    hour = group['time_point'].iloc[0]

    pivot_mean_radius = group.pivot(index='n_scales', columns='n_octaves', values='mean_roi_radius')
    pivot_n_sift = group.pivot(index='n_scales', columns='n_octaves', values='n_sift_descriptors')

    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    sns.heatmap(pivot_mean_radius, annot=True, fmt=".1f", cmap='viridis', ax=ax[0])
    ax[0].set_title(f'Mean ROI Radius\nWell: {well}, Time: {hour}, Center: {roi_center}')
    ax[0].set_xlabel('Number of Octaves')
    ax[0].set_ylabel('Number of Scales')

    sns.heatmap(pivot_n_sift, annot=True, fmt=".0f", cmap='magma', ax=ax[1])
    ax[1].set_title(f'Number of SIFT Descriptors\nWell: {well}, Time: {hour}, Center: {roi_center}')
    ax[1].set_xlabel('Number of Octaves')
    ax[1].set_ylabel('Number of Scales')

    plt.tight_layout()
    plt.show()
```

```python
# within each ROI, compute the percent change in number of SIFT descriptors between relative to n_octaves=8 and n_scales=3 (default parameters)
# then average the percent change of each combination of n_octaves and n_scales across all ROIs, resulting in a single heatmap

def compute_pct_change_gridsearch(summary_df, target_var='n_sift_descriptors'):
    ''' 
    For a given target variable in the summary_df, compute the percent change relative to the default SIFT parameters (n_octaves=8, n_scales=3) for each ROI.
    Then average the percent change across all ROIs and return a pivot table of the average percent change.
    '''
    summary_df_default = summary_df[(summary_df['n_octaves'] == 8) & (summary_df['n_scales'] == 3)][['roi_center', target_var]]
    summary_df_default = summary_df_default.rename(columns={target_var: f'default_{target_var}'})
    summary_df = summary_df.merge(summary_df_default, on='roi_center')
    summary_df['percent_change'] = (summary_df[target_var] - summary_df[f'default_{target_var}']) / summary_df[f'default_{target_var}'] * 100
    
    percent_change_summary = []
    for roi_center, group in summary_df.groupby('roi_center'):
        pivot = group.pivot(index='n_scales', columns='n_octaves', values='percent_change')
        percent_change_summary.append(pivot)

    # take the average percent change across all ROIs
    avg_percent_change = sum(percent_change_summary) / len(percent_change_summary)
    return avg_percent_change
```

```python
n_descriptors_pct_change = compute_pct_change_gridsearch(summary_df, target_var='n_sift_descriptors')
n_descriptors_pct_change
```

```python
mean_radius_pct_change = compute_pct_change_gridsearch(summary_df, target_var='mean_roi_radius')
mean_radius_pct_change
```

```python
# plot heatmaps for the pct change dataframes
fig, ax = plt.subplots(2, 1, figsize=(2,4), tight_layout=True, sharex=True)

sns.heatmap(n_descriptors_pct_change, annot=True, fmt=".1f", cmap='coolwarm', center=0, vmin=-100, vmax=100, ax=ax[0], cbar=False)
ax[0].set_title('Number of keypoints\n% change from default')
ax[0].set_xlabel('Number of octaves')
ax[0].set_ylabel('Number of scales')

sns.heatmap(mean_radius_pct_change, annot=True, fmt=".1f", cmap='coolwarm', center=0, vmin=-100, vmax=100, ax=ax[1], cbar=False)
ax[1].set_title('Mean keypoint radius\n% change from default')
ax[1].set_xlabel('Number of octaves')
ax[1].set_ylabel('Number of scales')

fig.savefig('figures/supp_figs/sift_gridsearch_pct_change_heatmaps.pdf', dpi=300, bbox_inches='tight')

plt.show()
```

```python
plot_df = summary_df.loc[summary_df['n_octaves'] == 8]

# make a line plot that shows the number of SIFT descriptors vs downsample factor for each ROI
fig, ax = plt.subplots(1, 2, figsize=(6,2), tight_layout=True)
for roi_center, group in summary_df.groupby('roi_center'):
    well = group['well'].iloc[0]
    hour = group['time_point'].iloc[0]
    # sort the group by n_scales
    group = group.sort_values(by='n_scales')
    ax[0].plot(group['n_scales'], group['n_sift_descriptors'], marker='o', label=f'Well {well}, {hour}h, {roi_center}')
    ax[1].plot(group['n_scales'], group['mean_roi_radius']*um_per_pixel, marker='o', label=f'Well {well}, {hour}h, {roi_center}')
ax[0].set_xlabel('Number of scales')
ax[0].set_ylabel('Number of keypoints detected')
ax[0].set_title('Number of keypoints vs scales')
# ax[0].legend(title='ROIs')

ax[1].set_xlabel('Number of scales')
ax[1].set_ylabel('Mean keypoint radius (um)')
ax[1].set_title('Mean radius vs scales')
ax[1].legend(title='ROIs', bbox_to_anchor=(1.05, 1), loc='upper left')

fig.savefig('figures/supp_figs/sift_keypoints_vs_scales.pdf', bbox_inches='tight', dpi=300)
plt.show()
```

```python

```
