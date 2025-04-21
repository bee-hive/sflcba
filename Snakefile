configfile: "config.yaml"

rule all:
    input:
        "analysis/adata_processed.h5ad"

rule compute_sift:
    input:
        ET_ratios = "data/4DonorAssay_ET_ratios.csv",
        RASA2KO_titrations = "data/4DonorAssay_RASA2KO_titrations.csv",
    output:
        "analysis/adata.h5ad"
    log:
        "logs/compute_sift.log"
    params:
        image_folder = config["image_folder_path"],
        downsample_pct = config["downsample_pct"],
        donor_ids = ",".join([str(x) for x in config["donor_ids"]]),
        well_id_rows = ",".join(config["well_id_rows"]),
        well_id_cols = ",".join(config["well_id_cols"]),
        rfp_threshold = config["rfp_threshold"],
    shell:
        '''
        python scripts/compute_sift_embedding.py \
            --ET_ratios {input.ET_ratios} \
            --RASA2KO_titrations {input.RASA2KO_titrations} \
            --image_folder {params.image_folder} \
            --downsample_pct {params.downsample_pct} \
            --donor_ids {params.donor_ids} \
            --well_id_rows {params.well_id_rows} \
            --well_id_cols {params.well_id_cols} \
            --rfp_threshold {params.rfp_threshold} \
            --output {output} \
            &> {log}
        '''

rule cluster_sift:
    input:
        "analysis/adata.h5ad"
    output:
        "analysis/adata_kmeans.h5ad"
    log:
        "logs/cluster_sift.log"
    params:
        k_values = ",".join([str(x) for x in config["k_values"]]),
    shell:
        '''
        python scripts/cluster_sift_embedding.py \
            --input {input} \
            --output {output} \
            --k_values {params.k_values} \
            &> {log}
        '''

rule glcm_and_rfp_stats:
    input:
        "analysis/adata_kmeans.h5ad"
    output:
        "analysis/adata_processed.h5ad"
    log:
        "logs/glcm_and_rfp_stats.log"
    shell:
        '''
        python scripts/glcm_and_rfp_stats.py \
            --input {input} \
            --output {output} \
            &> {log}
        '''