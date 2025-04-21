#!/usr/bin/env python
import argparse
import tiffile as tiff
import cv2 as cv
import anndata as ad
from sflcba.glcm import *

def parse_args():
    parser = argparse.ArgumentParser(description="Compute SIFT embedding for images.")
    parser.add_argument('--input', required=True, help="AnnData object after clustering (e.g. analysis/adata_kmeans.h5ad)")
    parser.add_argument('--output', required=True, help="Output path for the AnnData file after adding GLCM and RFP stats to adata.obs (e.g. analysis/adata_processed.h5ad)")
    return parser.parse_args()

def load_image(row, rfp=False):
    """
    Load an image based on the filename provided in the row.
    If rfp is True, load the corresponding RFP channel image.
    Note that one might have to change the file path for the red images depending the file structure of the data.
    """
    if rfp:
        filename = row['filename'].replace('phase_registered', 'red_registered')
        image = tiff.imread(filename)
    else:
        filename = row['filename']
        image = tiff.imread(filename)
        # Only normalize the brightfield image to the range [0, 255]
        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    return image

def main():
    args = parse_args()
    
    # load the adata object
    adata = ad.read_h5ad(args.input)

    # reset the index of the obs dataframe
    adata.obs = adata.obs.reset_index(drop=True)

    # ========================================================
    # Main loop: Process each image file (grouped by filename)
    # ========================================================
    for bf_path, image_df in adata.obs.groupby('filename'):

        # Load the brightfield (BF) and RFP images only once for this group.
        bf_image = load_image(image_df.iloc[0])
        rfp_image = load_image(image_df.iloc[0], rfp=True)
        
        # Compute mean RFP intensities for all ROIs in this image.
        mean_intensities = compute_mean_intensities(rfp_image, image_df)

        # compute fraction of RFP positive pixels for all ROIs in this image.
        rfp_pos_fractions = compute_rfp_pos_fractions(rfp_image, image_df)
        
        # Compute RFP Moran's I for all ROIs in this image.
        rfp_morans_Is = compute_morans_Is(rfp_image, image_df)

        # Compute GLCM stats for all ROIs in this image.
        glcm_homogeneity, glcm_energy = compute_glcm(bf_image, image_df)

        # Compute brightfield Moran's I for all ROIs in this image.
        bf_morans_Is = compute_morans_Is(bf_image, image_df)
        
        # Update the main DataFrame using .loc with the image_df indices.
        adata.obs.loc[image_df.index, 'roi_mean_rfp_intensity'] = mean_intensities
        adata.obs.loc[image_df.index, 'roi_rfp_pos_frac'] = rfp_pos_fractions
        adata.obs.loc[image_df.index, 'roi_rfp_morans_I'] = rfp_morans_Is
        adata.obs.loc[image_df.index, 'roi_glcm_homogeneity'] = glcm_homogeneity
        adata.obs.loc[image_df.index, 'roi_glcm_energy'] = glcm_energy
        adata.obs.loc[image_df.index, 'roi_bf_morans_I'] = bf_morans_Is

    # save the adata object
    adata.write(args.output)

if __name__ == "__main__":
    main()