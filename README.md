# saft_figuren

Code coresponding to the segmentation free live-cell behavioral analysis (SF-LCBA) manuscript.

### Installing dependencies

To reproduce this code, we recommend intalling all dependencies into a miniforge conda environment using the instructions as follows.

```
conda create --name saft python=3.11.4
conda activate saft
pip install -r requirements.txt
```

Then install the `sflcba` package in editable mode

```
pip install -e .
```
This command installs `sflcba` as a development (editable) package, meaning that any changes you make to the code in the sflcba folder will be immediately reflected without needing to reinstall the package.

After installing these dependencies, you should be able to lauch jupyter from the `saft` conda environment and run the corresponding notebooks in this repository.


### General usage

All core functions of SF-LCBA are found within the `sflcba` package. This package is split into subpackages for each section of the manuscript. Individual functions from these subpackages can be imported as long as the `saft` conda environment is activated, regardless of whether the python file is located in the `saft_figuren` folder or not.

```
from sflcba.utils import *
from sflcba.sift import compute_sift_embedding
```


### Repeating analysis from the manuscript

Repeating analysis shown in the manuscript starts by running a snakemake pipeline defined by `Snakefile`. This pipeline is responsible for
1. Thresholding the red channel into a binary mask
2. Computing the spatial entropy of the red channel's binary mask
3. Running SIFT on phase images
4. Using the SIFT descriptor vectors to construct an embedding. Note that the total number of rows in this embedding is scaled by the true number of keypoints in all images multiplied by `downsample_pct` (specified in `config.yaml`).
5. Clustering the embedding using k-means.

Before running the pipeline, make sure to set the `image_folder_path` in `config.yaml` to the path to the folder containing the images. This folder is expected to have the following structure:

```
image_folder_path
├── Donor1
│   ├── phase_registered
|   |   ├── {well_id}_*_{time_point}.tif
|   |   ├── {well_id}_*_{time_point}.tif
│   |   ├── ...
│   ├── red_registered
|   |   ├── {well_id}_*_{time_point}.tif
|   |   ├── {well_id}_*_{time_point}.tif
│   |   ├── ...
├── Donor2
│   ├── phase_registered
|   |   ├── {well_id}_*_{time_point}.tif
|   |   ├── {well_id}_*_{time_point}.tif
│   |   ├── ...
│   ├── red_registered
|   |   ├── {well_id}_*_{time_point}.tif
|   |   ├── {well_id}_*_{time_point}.tif
│   |   ├── ...
├── Donor3
│   ├── ...
...
```

Where `{well_id}` is one of the well ids in `well_id_rows` and `well_id_cols` in `config.yaml`, and `{time_point}` is one of the time points in `time_points` in `config.yaml`. The `donor_ids` list in `config.yaml` specifies the suffix ID of all donors to process. An example tif file name would be `Donor2/phase_registered/E5_reg_0003.tif` which would correspond to the phase image from donor 2, well_id E5 and time point 3. All time points must be zero-padded in the file name to the same number of digits to allow for sorting.

Additional parameters in the `config.yaml` that must be set by the user are `rfp_threshold`, which sepcifies the intensity threshold for the red channel, and `k_values`, which specifies the k values to use for k-means clustering.

The pipeline is run using the command `snakemake --use-conda --rerun-incomplete --keep-going`

Running the pipeline will produce several files in an `analysis/` folder. These files are then used as input for jupyter notebooks in the `notebooks/` folder. Running the jupyter notebooks will reproduce all the figures, tables, and statistical tests shown in the manuscript. The notebooks are uploaded to this repository in markdown format to allow for easy version control. If you wish to run the notebooks yourself, you will first need to convert the markdown files to .ipynb files with no output by running

```
jupytext --to notebook notebooks/*.md
```

Keep in mind that the `saft` conda environment must be activated before running both the snakemake pipeline and the jupyter notebooks.
