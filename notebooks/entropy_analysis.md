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
# adata = ad.read_h5ad('analysis/adata_20250225_processed_20250310.h5ad')
adata = ad.read_h5ad('analysis/adata_processed.h5ad')
adata
```

```python
# move PC1 and PC2 from adata.obsm['X_pca'] to adata.obs['PC1'] and adata.obs['PC2']
adata.obs['PC1'] = adata.obsm['X_pca'][:, 0]
adata.obs['PC2'] = adata.obsm['X_pca'][:, 1]
```

```python
# compute a replicate_id integer based on the well_id
# if the well_id ends in an odd number, then the replicate_id is 1
# if the well_id ends in an even number, then the replicate_id is 0
adata.obs['replicate_id'] = adata.obs['well_id'].apply(lambda x: int(x[-1]) % 2)
```

```python
adata.obs.head()
```

```python
adata.obs.columns
```

```python
entropy_df = adata.obs[['donor_id', 'time', 'well_id', 'replicate_id','rasa2ko_titration', 'et_ratio', 'entropy', 'p_areas', 'n_og_keypoints']].drop_duplicates()
entropy_df
```

```python
# plot a histogram of the number of keypoints per image
fig, ax = plt.subplots(1, 1, figsize=(2, 2))
entropy_df['n_og_keypoints'].hist(bins=50, ax=ax)
ax.set_xlabel('# SIFT keypoints per image (d_n)')
ax.set_ylabel('Count')
ax.set_title('N={} images'.format(len(entropy_df)))
sns.despine(ax=ax)
# fig.savefig('figures/fig2/n_keypoints_hist.pdf', bbox_inches='tight', dpi=300)
plt.show()
```

```python
g = sns.PairGrid(entropy_df[['entropy', 'p_areas', 'n_og_keypoints', 'et_ratio', 'rasa2ko_titration', 'time']], diag_sharey=False, hue='time')
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot, s=1, alpha=0.1)
plt.show()
```

```python
# plot a scatterplot of entropy vs p_areas with the hue being time
# alongside the same scatterplot with the hue being et_ratio
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax = ax.flatten()
sns.scatterplot(ax=ax[0], data=entropy_df, x='p_areas', y='entropy', hue='time', s=1, alpha=0.5)
sns.scatterplot(ax=ax[1], data=entropy_df, x='p_areas', y='entropy', hue='et_ratio', s=1, alpha=0.5)
sns.scatterplot(ax=ax[2], data=entropy_df, x='p_areas', y='entropy', hue='rasa2ko_titration', s=1, alpha=0.5)

for i in range(3):
    ax[i].set_xlabel('RFP+ area')
    ax[i].set_ylabel('Entropy')
    ax[i].set_title('Cancer cell entropy vs area per image')
    # log scale the x axis
    # ax[i].set_xscale('log')

plt.show()
```

## Look at the entropy values over time for different RASA2KO titrations and E:T ratios

```python


def histogram_intersection(hist1, hist2):
    return np.sum(np.minimum(hist1, hist2))
    
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def plot_entropy_vs_time_confidence_interval(entropy_df, column, ax=None,confidence=0.95, max_time=np.inf, 
                                             x_col='time',
                                             y_col='entropy',
                                             title="Entropy of RFP cancer cell mask",
                                             x_label="Time (frame number)",
                                             y_label="Spatial entropy (95% confidence interval)",):
    
    if ax is None:
        fig, ax = plt.subplots()
    
    for value, df in entropy_df.groupby(column):
        # find the unique times, sort them, and take only the times that are less than max_time
        unique_times = df[x_col].unique()
        unique_times = np.sort(unique_times)
        unique_times = unique_times[unique_times <= max_time]

        # placeholder for the mean, lower bound, and upper bound
        interval_holder0 = []

        for t in unique_times:
            # find the mean and the confidence interval for each time
            temp_df = df[df[x_col] == t]
            mean0, blb0, bub0 = mean_confidence_interval(temp_df[y_col], confidence=confidence)
            interval0 = [mean0, blb0, bub0]
            interval_holder0.append(interval0)

        # convert the placeholder to an array
        interval_holder0 = np.array(interval_holder0)

        # plot the confidence interval
        ax.fill_between(unique_times, interval_holder0[:,1], interval_holder0[:,2], alpha=0.6, label=value)

    sns.despine(ax=ax)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.legend(title=column)

fig, ax = plt.subplots(1, 2, figsize=(8,4), tight_layout=True)
ax = ax.flatten()
plot_entropy_vs_time_confidence_interval(entropy_df, 'rasa2ko_titration', ax=ax[0], confidence=0.95, max_time=64)
plot_entropy_vs_time_confidence_interval(entropy_df, 'et_ratio', ax=ax[1], confidence=0.95, max_time=64)
ax[0].legend(title='RASA2KO titration (%)', loc='lower left')
ax[1].legend(title='E:T ratio', loc='lower left')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(8,4), tight_layout=True)
ax = ax.flatten()
plot_entropy_vs_time_confidence_interval(entropy_df, 'rasa2ko_titration', ax=ax[0], confidence=0.95, max_time=64, y_col='p_areas', title='Area of RFP cancer cell mask', y_label='# of RFP+ pixels per frame (95% confidence interval)')
plot_entropy_vs_time_confidence_interval(entropy_df, 'et_ratio', ax=ax[1], confidence=0.95, max_time=64, y_col='p_areas', title='Area of RFP cancer cell mask', y_label='# of RFP+ pixels per frame (95% confidence interval)')
ax[0].legend(title='RASA2KO titration (%)', loc='upper left')
ax[1].legend(title='E:T ratio', loc='upper left')
plt.show()

```

```python
fig, ax = plt.subplots(2, 1, figsize=(2.5,4), tight_layout=True, sharex=True)
ax = ax.flatten()

plot_entropy_vs_time_confidence_interval(entropy_df, 'rasa2ko_titration', ax=ax[0], confidence=0.95, max_time=64, y_label="Spatial entropy (95% CI)", x_label="Time")
plot_entropy_vs_time_confidence_interval(entropy_df, 'et_ratio', ax=ax[1], confidence=0.95, max_time=64, y_label="Spatial entropy (95% CI)", x_label="Time")
ax[0].legend(title='RASA2KO', loc='lower left')
ax[1].legend(title='E:T', loc='lower left')

# # set the y-axis limits to be the same for all subplots
# for i in range(2):
#     ax[i].set_ylim(3.5, 4.5)

# fig.savefig('figures/fig1/entropy_vs_time.pdf', bbox_inches='tight', dpi=200)

plt.show()
```

```python
red_path = adata.obs['filename'][0].replace('phase_registered', 'red_registered')

resized_latish_red = tiff.imread(red_path)
# threshold the red channel into a binary mask
aggregate_threshed = resized_latish_red > 3.5

# check if red_path file exists
if not os.path.exists(red_path):
    print('File does not exist: {}'.format(red_path))
else:
    print('File exists: {}'.format(red_path))
```

```python
entropy_df[(entropy_df['rasa2ko_titration'] == 100) & (entropy_df['et_ratio'] == 2.8284) & (entropy_df['replicate_id'] == 0)]['well_id'].unique()[0]
```

```python
def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def plot_representative_fn(metadata_df, rasa_value, et_value, replicate_id, donor_id, time_point, trim=100, lab_folder_path='/gladstone/engelhardt/lab/', ax=None, phase=True):
    '''
    Given a rasa value, et value, replicate id, donor id, and time point, plot the representative bright field + RFP image. 
    The mapping of rasa_value, et_value, and replicate_id to well_id is done using metadata_df which is a DataFrame with columns 'rasa2ko_titration', 'et_ratio', 'replicate_id', and 'well_id'.
    Overlay the RFP mask onto the bright field image if phase=True.
    '''
    # create a list of all the image file paths that correspond to this donor_id
    donor_location_phase = lab_folder_path + "MarsonLabIncucyteData/AnalysisFiles/4DonorAssay/registered_images/Donor{}/phase_registered/*tif".format(donor_id)
    donor_location_red = lab_folder_path + "MarsonLabIncucyteData/AnalysisFiles/4DonorAssay/registered_images/Donor{}/red_registered/*tif".format(donor_id)
    files_phase = glob.glob(donor_location_phase)
    files_red = glob.glob(donor_location_red)

    # find the well id that corresponds to the specified rasa_value, et_value, and replicate_id
    well_id = metadata_df[(metadata_df['rasa2ko_titration'] == rasa_value) & (metadata_df['et_ratio'] == et_value) & (metadata_df['replicate_id'] == replicate_id)]['well_id'].unique()[0]

    # find the list of file paths that correspond to the specified well_id
    # store all image paths from this well in a list, sorted by timepoint
    #phase
    matching = [s for s in files_phase if (well_id + "_") in s]
    sorted_file_list_phase = (sorted_nicely(matching))
    # #red
    matching = [s for s in files_red if (well_id + "_") in s]
    sorted_file_list_red = (sorted_nicely(matching))

    # subset the list to the specified time point and load the images
    resized_latish_phase = tiff.imread(sorted_file_list_phase[time_point])
    resized_latish_red = tiff.imread(sorted_file_list_red[time_point])
    # threshold the red channel into a binary mask
    red_frame = resized_latish_red > 3.5

    # normalize intensity of the phase image
    phase_frame = cv.normalize(resized_latish_phase, None, 0, 255, cv.NORM_MINMAX).astype('uint8') 

    # plot the phase image with the red mask superimposed
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6,6))

    trim=100
    ax.imshow(red_frame[trim:-trim,trim:-trim], cmap='Reds', alpha = 1.0)
    if phase:
        ax.imshow(phase_frame[trim:-trim,trim:-trim], cmap='gray', alpha = .75)
    ax.set_title('RASA2KO={}%\nE:T={}, t={}'.format(round(rasa_value,1),round(et_value,2),time_point))
    # remove the axes ticks and labels
    ax.set_axis_off()
    
    return


plot_representative_fn(entropy_df, rasa_value=100, et_value=1.0, replicate_id=0, donor_id=1, time_point=0, trim=100, phase=False)

```

```python
fig, ax = plt.subplots(4, 5, figsize=(5,5), tight_layout=True)

# find 5 evenly spaced time points between 0 and 64
times = [0, 15, 30, 45, 60]
# donor ID is always 1 and replicate ID is always 0
donor_id = 1
replicate_id = 0
phase=False

# top row of subplots have rasa_value=50, et_value=2.0
for i in range(5):
    plot_representative_fn(entropy_df, rasa_value=50, et_value=2.0, replicate_id=replicate_id, donor_id=donor_id, time_point=times[i], trim=100, ax=ax[0,i], phase=phase)

# second row of subplots have rasa_value=50, et_value=1.0
for i in range(5):
    plot_representative_fn(entropy_df, rasa_value=50, et_value=1.0, replicate_id=replicate_id, donor_id=donor_id, time_point=times[i], trim=100, ax=ax[1,i], phase=phase)

# third row of subplots have rasa_value=100, et_value=2.0
for i in range(5):
    plot_representative_fn(entropy_df, rasa_value=100, et_value=2.0, replicate_id=replicate_id, donor_id=donor_id, time_point=times[i], trim=100, ax=ax[2,i], phase=phase)

# fourth row of subplots have rasa_value=100, et_value=1.0
for i in range(5):
    plot_representative_fn(entropy_df, rasa_value=100, et_value=1.0, replicate_id=replicate_id, donor_id=donor_id, time_point=times[i], trim=100, ax=ax[3,i], phase=phase)


# fig.savefig('figures/fig1/rfp_mask_grid.pdf', bbox_inches='tight', dpi=400)

plt.show()
```

```python
# repeat the above plot showing both phase and red channel
fig, ax = plt.subplots(4, 5, figsize=(8,8), tight_layout=True)

# find 5 evenly spaced time points between 0 and 64
times = [0, 15, 30, 45, 60]
# donor ID is always 1 and replicate ID is always 0
donor_id = 1
replicate_id = 0
phase=True

# top row of subplots have rasa_value=50, et_value=2.0
for i in range(5):
    plot_representative_fn(entropy_df, rasa_value=50, et_value=2.0, replicate_id=replicate_id, donor_id=donor_id, time_point=times[i], trim=100, ax=ax[0,i], phase=phase)

# second row of subplots have rasa_value=50, et_value=1.0
for i in range(5):
    plot_representative_fn(entropy_df, rasa_value=50, et_value=1.0, replicate_id=replicate_id, donor_id=donor_id, time_point=times[i], trim=100, ax=ax[1,i], phase=phase)

# third row of subplots have rasa_value=100, et_value=2.0
for i in range(5):
    plot_representative_fn(entropy_df, rasa_value=100, et_value=2.0, replicate_id=replicate_id, donor_id=donor_id, time_point=times[i], trim=100, ax=ax[2,i], phase=phase)

# fourth row of subplots have rasa_value=100, et_value=1.0
for i in range(5):
    plot_representative_fn(entropy_df, rasa_value=100, et_value=1.0, replicate_id=replicate_id, donor_id=donor_id, time_point=times[i], trim=100, ax=ax[3,i], phase=phase)

# fig.savefig('figures/fig1/rfp_phase_grid.pdf', bbox_inches='tight', dpi=400)

plt.show()
```

```python
# sweep over RASA2KO titration at a constant E:T ratio
fig, ax = plt.subplots(5, 5, figsize=(8,9.5), tight_layout=True)


# find 5 evenly spaced time points between 0 and 64
times = [0, 15, 30, 45, 60]
# donor ID is always 1 and replicate ID is always 0
donor_id = 1
replicate_id = 0
# only show the RFP channel
phase=False
# set E:T ratio constant at 1.0
# 
et_value = 1.0

rasa_values = np.sort(entropy_df['rasa2ko_titration'].unique())

for i in range(len(rasa_values)):
    rasa_value = rasa_values[i]
    # top row of subplots have rasa_value=50, et_value=2.0
    for j in range(5):
        plot_representative_fn(entropy_df, rasa_value=rasa_value, et_value=et_value, replicate_id=replicate_id, donor_id=donor_id, time_point=times[j], trim=100, ax=ax[i,j], phase=phase)

# fig.savefig('figures/fig1/rfp_mask_rasa2ko_grid.pdf', bbox_inches='tight', dpi=400)

plt.show()
```

```python
# sweep over E:T ratio at a constant RASA2KO titration
fig, ax = plt.subplots(5, 5, figsize=(8,9.5), tight_layout=True)

# find 5 evenly spaced time points between 0 and 64
times = [0, 15, 30, 45, 60]
# donor ID is always 1 and replicate ID is always 0
donor_id = 1
replicate_id = 0
# only show the RFP channel
phase = False
# set RASA2KO titration constant at 50
rasa_value = 50

et_values = np.sort(entropy_df['et_ratio'].unique())

for i in range(len(et_values)):
    et_value = et_values[i]
    for j in range(5):
        plot_representative_fn(entropy_df, rasa_value=rasa_value, et_value=et_value, replicate_id=replicate_id, donor_id=donor_id, time_point=times[j], trim=100, ax=ax[i,j], phase=phase)

# fig.savefig('figures/fig1/rfp_mask_et_grid.pdf', bbox_inches='tight', dpi=400)

plt.show()
```

### Use general linear model to regress entropy against all the covariates

```python
def get_coef_df(model):
    '''
    Returns a dataframe of coefficients from a statsmodels linear regression model.
    Also returns the p-values and the adjusted p-values using the Benjamini-Hochberg method.
    '''
    lin_reg = model
    err_series = lin_reg.params - lin_reg.conf_int()[0]
    coef_df = pd.DataFrame({'coef': lin_reg.params.values[1:-1],
                            'err': err_series.values[1:-1],
                            'varname': err_series.index.values[1:-1],
                            'pvalue': lin_reg.pvalues.values[1:-1]
                           })
    # Apply Benjamini-Hochberg FDR correction
    coef_df['p_adj'] = multipletests(coef_df['pvalue'], method='fdr_bh')[1]
    
    return coef_df


def coef_plot(coef_df, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    
    coef_df.plot(x='varname', y='coef', kind='bar', 
                 ax=ax, color='none', 
                 yerr='err', legend=False)
    ax.set_ylabel('Coefficient (mean +/- std err)')
    ax.set_xlabel('MixedLM term')
    ax.scatter(x=np.arange(coef_df.shape[0]), 
               marker='s', s=20, 
               y=coef_df['coef'], color='black')
    ax.axhline(y=0, linestyle='--', color='grey', linewidth=1)
    # ax.xaxis.set_ticks_position('none')

    # annotate p-values in scientific notation
    for i, pval in enumerate(coef_df['p_adj']):
        if pval > 0.05:
            continue
        elif pval > 0.01:
            ax.text(i, coef_df['coef'].max() + 0.005, 'p={:.1e}'.format(pval), va='bottom', ha='center', rotation=90, color='k')
        elif pval > 0.001:
            ax.text(i, coef_df['coef'].max() + 0.005, 'p={:.1e}'.format(pval), va='bottom', ha='center', rotation=90, color='orange')
        else:
            ax.text(i, coef_df['coef'].max() + 0.005, 'p={:.1e}'.format(pval), va='bottom', ha='center', rotation=90, color='red')
    
    sns.despine(ax=ax, trim=True)

```

```python
temp_df = entropy_df.rename(columns={'rasa2ko_titration': 'RASA2', 'et_ratio': 'ET', 'replicate_id': 'replicate', 'donor_id': 'donor', 'p_areas': 'RFP_area', 'n_og_keypoints': 'n_keypoints'})

formula1 = "entropy ~ RASA2 * ET * time * C(replicate)"
mdf1 = smf.mixedlm(formula1,
                 temp_df,
                 groups=temp_df["donor"],
                 )
mdf1 = mdf1.fit(reml=False)
print(mdf1.summary())
fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
coef_df1 = get_coef_df(mdf1)
coef_plot(coef_df1, ax=ax)
ax.set_title(formula1)

# fig.savefig('figures/fig1/entropy_mixedlm_coef.pdf', bbox_inches='tight')

plt.show()

```

```python
coef_df1
```

```python
formula2 = "RFP_area ~ RASA2 * ET * time * C(replicate)"
mdf2 = smf.mixedlm(formula2,
                 temp_df,
                 groups=temp_df["donor"],
                 )
mdf2 = mdf2.fit(reml=False)
print(mdf2.summary())
fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
coef_df2 = get_coef_df(mdf2)
coef_plot(coef_df2, ax=ax)
ax.set_title(formula2)

# fig.savefig('figures/fig1/rfp_area_mixedlm_coef.pdf', bbox_inches='tight')

plt.show()
```

```python
coef_df2
```

```python

formula3 = "n_keypoints ~ RASA2 * ET * time * C(replicate)"
mdf3 = smf.mixedlm(formula3,
                 temp_df,
                 groups=temp_df["donor"],
                 )
mdf3 = mdf3.fit(reml=False)
print(mdf3.summary())
fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
coef_df3 = get_coef_df(mdf3)
coef_plot(coef_df3, ax=ax)
ax.set_title(formula3)

# fig.savefig('figures/fig1/n_keypoints_mixedlm_coef.pdf', bbox_inches='tight')

plt.show()
```

```python
coef_df3
```

```python
# create a scatter plot of the coefficients for each term in mdf2 (RFP area as target variable) vs mdf3 (number of keypoints as target variable)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
coeffs2 = mdf2.params.values[1:-1]
errs2 = coeffs2 - mdf2.conf_int()[0][1:-1]
coeffs3 = mdf3.params.values[1:-1]
errs3 = coeffs3 - mdf3.conf_int()[0][1:-1]
# sns.scatterplot(x=coeffs2, y=coeffs3, ax=ax)
ax.errorbar(x=coeffs2, y=coeffs3, xerr=errs2, yerr=errs3, capsize=3, color='black', fmt='o', alpha=0.5)
# annotate with seaborn line of best fit
sns.regplot(x=coeffs2, y=coeffs3, ax=ax, line_kws={'color': 'grey', 'lw': 1, 'ls': '--'})
ax.set_xlabel('RFP area coefficient')
ax.set_ylabel('Number of keypoints coefficient')
sns.despine(ax=ax)
plt.show()
```

### Make a multipanel supplemental figure

```python
fig, ax = plt.subplots(2, 3, figsize=(8, 8), tight_layout=True)
ax = ax.flatten()

# plot RFP area over time in the first two subplots
plot_entropy_vs_time_confidence_interval(entropy_df, 'rasa2ko_titration', ax=ax[0], confidence=0.95, max_time=64, y_col='p_areas', title='Area of RFP cancer cell mask', y_label='# of RFP+ pixels per frame (95% confidence interval)')
plot_entropy_vs_time_confidence_interval(entropy_df, 'et_ratio', ax=ax[1], confidence=0.95, max_time=64, y_col='p_areas', title='Area of RFP cancer cell mask', y_label='# of RFP+ pixels per frame (95% confidence interval)')
ax[0].legend(title='RASA2KO\ntitration (%)', loc='upper left')
ax[1].legend(title='E:T ratio', loc='upper left')

# plot the regression coefficients of RFP area in the top right subplot
coef_plot(coef_df2, ax=ax[2])
ax[2].set_title(formula2)

# plot a scatterplot of entropy vs p_areas with the hues being time, et_ratio, and rasa2ko_titration
sns.scatterplot(ax=ax[3], data=entropy_df, x='p_areas', y='entropy', hue='time', s=1, alpha=0.5, rasterized=True)
sns.scatterplot(ax=ax[4], data=entropy_df, x='p_areas', y='entropy', hue='et_ratio', s=1, alpha=0.5, rasterized=True)
sns.scatterplot(ax=ax[5], data=entropy_df, x='p_areas', y='entropy', hue='rasa2ko_titration', s=1, alpha=0.5, rasterized=True)

for i in range(3,6):
    ax[i].set_xlabel('RFP+ area (# pixels)')
    ax[i].set_ylabel('RFP spatial entropy')
    ax[i].set_title('Cancer cell entropy vs area per image')
    sns.despine(ax=ax[i])

# fig.savefig('figures/fig1/rfp_vs_entropy_multipanel.pdf', bbox_inches='tight', dpi=400)

plt.show()
```

```python

```
