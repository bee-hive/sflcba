#!/usr/bin/env python
# scripts/compute_sift_embedding.py
import os
import yaml
import anndata as ad
import numpy as np
import sflcba.sift as sift
import glob

def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    lab_folder = config["lab_folder_path"]
    downsample_pct = config["downsample_pct"]
    donor_ids = config["donor_ids"]
    well_ids = config["well_ids"]

    adata_list = []

    # For each donor and well, process images (this is a simplified example)
    for d in donor_ids:
        # Construct file search pattern (modify as needed)
        phase_pattern = os.path.join(lab_folder, f"MarsonLabIncucyteData/AnalysisFiles/4DonorAssay/registered_images/Donor{d}/phase_registered/*.tif")
        files = sorted(glob.glob(phase_pattern))
        for well in well_ids:
            # Filter files for this well (assumes well id appears in filename)
            well_files = [f for f in files if well in f]
            for image_path in well_files:
                embed = sift.compute_sift_embedding(image_path, downsample_pct=downsample_pct)
                # Here, you would package embed and related metadata into an AnnData object.
                # For demonstration we assume embed["descriptors"] exists.
                if embed["descriptors"].size == 0:
                    continue
                adata = ad.AnnData(X=embed["descriptors"])
                # (Additional metadata such as donor_id, well_id, etc. can be added to adata.obs)
                adata.obs["donor_id"] = d
                adata.obs["well_id"] = well
                adata_list.append(adata)

    # Concatenate all AnnData objects and save to file
    if adata_list:
        adata_all = ad.concat(adata_list, join="outer")
    else:
        adata_all = ad.AnnData()
    os.makedirs("analysis", exist_ok=True)
    adata_all.write("analysis/adata.h5ad")

if __name__ == "__main__":
    main()
