#!/usr/bin/env python
import argparse
import anndata as ad
import sflcba.cluster as cluster

def main():
    parser = argparse.ArgumentParser(description="Cluster SIFT embeddings using k-means.")
    parser.add_argument('--input', required=True, help="Path to the input AnnData file (e.g. analysis/adata.h5ad)")
    parser.add_argument('--output', required=True, help="Path to save the output AnnData file (e.g. analysis/adata_kmeans.h5ad)")
    parser.add_argument('--k_values', default="3,4,5,6,7,8,9,10", help="Comma-separated list of k values (default: '3,4,5,6,7,8,9,10')")
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input)
    k_values = [int(x) for x in args.k_values.split(",")]
    adata = cluster.cluster_sift_embedding(adata, k_values=k_values)
    adata.write(args.output)

if __name__ == "__main__":
    main()
