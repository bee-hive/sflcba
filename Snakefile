configfile: "config.yaml"

rule all:
    input:
        "analysis/adata_kmeans.h5ad"

rule compute_sift:
    output:
        "analysis/adata.h5ad"
    params:
        lab_folder = config["lab_folder_path"],
        downsample_pct = config["downsample_pct"],
        donor_ids = ",".join([str(x) for x in config["donor_ids"]]),
        well_ids = ",".join(config["well_ids"])
    shell:
        '''
        python scripts/compute_sift_embedding.py \
            --lab_folder {params.lab_folder} \
            --downsample_pct {params.downsample_pct} \
            --donor_ids {params.donor_ids} \
            --well_ids {params.well_ids} \
            --output {output}
        '''

rule cluster_sift:
    input:
        "analysis/adata.h5ad"
    output:
        "analysis/adata_kmeans.h5ad"
    params:
        k_values = "3,4,5,6,7,8,9,10"
    shell:
        '''
        python scripts/cluster_sift_embedding.py \
            --input {input} \
            --output {output} \
            --k_values {params.k_values}
        '''

