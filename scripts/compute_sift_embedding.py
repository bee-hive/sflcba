#!/usr/bin/env python
import os
import argparse
import itertools
import anndata as ad
import numpy as np
import pandas as pd
from sflcba.sift import compute_sift_embedding
from sflcba.utils import sorted_nicely, load_binary_image
from sflcba.entropy import entropy
import glob


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SIFT embedding for images.")
    parser.add_argument('--ET_ratios', required=True, help="csv file for plate map where the values are ET ratios")
    parser.add_argument('--RASA2KO_titrations', required=True, help="csv file for plate map where the values are RASA2KO titrations")
    parser.add_argument('--image_folder', required=True, help="Base lab folder path")
    parser.add_argument('--downsample_pct', type=float, required=True, help="Downsample percentage for SIFT keypoints")
    parser.add_argument('--donor_ids', required=True, help="Comma-separated list of donor IDs (e.g. '1,2,3,4')")
    parser.add_argument('--well_id_rows', required=True, help="Comma-separated list of well rows (e.g. 'B,D,E,F')")
    parser.add_argument('--well_id_cols', required=True, help="Comma-separated list of well columns (e.g. '1,2,3,4')")
    parser.add_argument('--rfp_threshold', type=float, default=3.5, help="Threshold for RFP channel")
    parser.add_argument('--output', required=True, help="Output path for the AnnData file (e.g. analysis/adata.h5ad)")
    return parser.parse_args()

def get_well_ids(well_id_rows, well_id_cols):
    """
    Generate well IDs based on the provided rows and columns.
    """
    well_ids = []
    for i in well_id_cols:
        rasas = [str(i)]
        combos = list(itertools.product(well_id_rows, rasas))
        newlist = ["".join(item) for item in combos]
        well_ids.extend(newlist)
    return well_ids


def map_well_to_experiment(well_id, rasa2ko_df, et_ratio_df):
    '''
    Given a well id, return the corresponding RASA2KO titration and E:T ratio.

    The first character of the well_id corresponds to the row of each dataframe.
    The second (and third) character(s) of the well_id corresponds to the column of each dataframe.
    '''
    row = well_id[0]
    col = well_id[1:]
    try:
        rasa2ko = rasa2ko_df.loc[row,col]
    except:
        print("No RASA2KO titration for well {}".format(well_id))
        rasa2ko = np.nan
    try:
        et_ratio = et_ratio_df.loc[row,col]
    except:
        print("No E:T ratio for well {}".format(well_id))
        et_ratio = np.nan
    return rasa2ko, et_ratio


def main():
    # Parse command line arguments.
    args = parse_args()

    image_folder = args.image_folder
    downsample_pct = args.downsample_pct
    donor_ids = [int(x) for x in args.donor_ids.split(",")]
   
    # Determine the well IDs based on the provided rows and columns.
    well_id_rows = args.well_id_rows.split(",")
    well_id_cols = args.well_id_cols.split(",")
    well_ids = get_well_ids(well_id_rows, well_id_cols)

    # Read the RASA2KO and E:T ratio plate map dataframes.
    rasa2ko_df = pd.read_csv(args.RASA2KO_titrations, index_col=0)
    et_ratio_df = pd.read_csv(args.ET_ratios, index_col=0)
    # convert rasa2ko_df entries of 'No TCR' and 'No T Cell' to NaN
    rasa2ko_df = rasa2ko_df.replace('No TCR ', np.nan)
    rasa2ko_df = rasa2ko_df.replace('No T Cell', np.nan)
    # change all columns of rasa2ko_df and et_ratio_df to float64
    rasa2ko_df = rasa2ko_df.astype('float64')
    et_ratio_df = et_ratio_df.astype('float64')


    adata_list = []
    # For each donor and well, process images.
    for d in donor_ids:
        phase_pattern = os.path.join(image_folder, f"Donor{d}/phase_registered/*.tif")
        red_pattern = os.path.join(image_folder, f"Donor{d}/red_registered/*.tif")
        files_phase = glob.glob(phase_pattern)
        files_red = glob.glob(red_pattern)
        for well in well_ids:
            # find all the file names for the red and phase channels of this donor and well
            # phase
            matching = [s for s in files_phase if (well + "_") in s]
            sorted_file_list_phase = (sorted_nicely(matching))
            # red
            matching = [s for s in files_red if (well + "_") in s]
            sorted_file_list_red = (sorted_nicely(matching))

            # get the RASA2KO and E:T ratio for this well
            rasa2ko, et_ratio = map_well_to_experiment(well, rasa2ko_df, et_ratio_df)

            max_times = len(sorted_file_list_phase)

            # loop through all the time points
            for t in range(max_times):
                # find the file names for the red and phase channels of this donor, well, and time
                phase_image_path = sorted_file_list_phase[t]
                red_image_path = sorted_file_list_red[t]

                # compute the entropy and area of the red mask
                red_frame = load_binary_image(red_image_path, threshold=args.rfp_threshold)
                # trim the red mask to remove the borders
                red_resized = red_frame[16:-16,16:-16]
                # compute the spatial entropy of the red mask
                red_entropy = np.sum(entropy(red_resized))
                # compute the RFP+ area of the red mask
                p_areas = np.sum((red_frame * 1).ravel())

                # compute the sift embedding for the phase image
                embed = compute_sift_embedding(phase_image_path, downsample_pct=downsample_pct)
                # Skip if no descriptors were found.
                if embed["descriptors"].size == 0:
                    continue
                adata = ad.AnnData(X=embed["descriptors"])

                # Attach metadata to the AnnData object.
                adata.obs['donor_id'] = d
                adata.obs['time'] = t
                adata.obs['well_id'] = well
                adata.obs['rasa2ko_titration'] = rasa2ko
                adata.obs['et_ratio'] = et_ratio
                adata.obs['entropy'] = red_entropy
                adata.obs['p_areas'] = p_areas
                adata.obs['filename'] = phase_image_path
                adata.obs['scales'] = embed["scales"]
                adata.obs['octaves'] = embed["octaves"]
                adata.obs['sigmas'] = embed["sigmas"]
                adata.obs['orientations'] = embed["orientations"]
                adata.obs['x'] = embed["x"]
                adata.obs['y'] = embed["y"]
                adata.obs['n_og_keypoints'] = embed["n_keypoints_original"]

                # add a sift_ prefix to the variable names
                adata.var_names = ['sift_{}'.format(i) for i in range(128)]

                adata_list.append(adata)

    # Concatenate all AnnData objects into a single object.
    if adata_list:
        adata_all = ad.concat(adata_list, join="outer")
    else:
        adata_all = ad.AnnData()
    
    # Save the combined AnnData object to the specified output file.
    adata_all.write(args.output)

if __name__ == "__main__":
    main()
