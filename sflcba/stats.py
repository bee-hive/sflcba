# sflcba/utils.py
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

### Function to compute Cohen's d (standardized effect size)
def cohens_d(x, y):
    """
    Calculate Cohen's d for two independent samples.
    """
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Input arrays must not be empty.")
    nx, ny = len(x), len(y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    std_x, std_y = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * std_x**2 + (ny - 1) * std_y**2) / (nx + ny - 2))
    return (mean_x - mean_y) / pooled_std if pooled_std > 0 else 0


def cluster_enrichment(df, cluster_column, continuous_vars, categorical_vars, method='fdr_bh'):
    """
    Compute p-values and adjusted p-values for continuous and categorical variables associated with cluster membership.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame where each row corresponds to a SIFT keypoint, columns respond to experimental metadata and cluster membership.
    cluster_column : str
        Column name for cluster membership in df.
    continuous_vars : list
        List of column names for continuous variables in df.
    categorical_vars : list
        List of column names for categorical variables in df.
    method : str, optional
        Method for multiple hypothesis correction. Default is 'fdr_bh'.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing p-values and adjusted p-values for continuous and categorical variables associated with cluster membership.
    """
    results = []
    # Iterate over each cluster
    for cluster in df[cluster_column].unique():
        cluster_mask = df[cluster_column] == cluster

        # 1. Kruskal-Wallis Test for Continuous Variables
        for var in continuous_vars:
            cluster_values = df.loc[cluster_mask, var]
            other_values = df.loc[~cluster_mask, var]

            if len(cluster_values) > 1 and len(other_values) > 1:  # Ensure enough data points
                stat, p_val = stats.kruskal(cluster_values, other_values)
                effect_size = cohens_d(cluster_values, other_values)  # Cohen's d

                results.append([cluster, var, None, p_val, effect_size, 'kruskal', 'cohens_d'])  # 'var_value' is None

        # 2. Chi-square Test for Categorical Variables
        for var in categorical_vars:
            for val in df[var].unique():
                in_cluster = sum((df[var] == val) & cluster_mask)
                out_cluster = sum((df[var] == val) & ~cluster_mask)
                not_in_cluster = sum(cluster_mask) - in_cluster
                not_out_cluster = sum(~cluster_mask) - out_cluster
                contingency_table = np.array([[in_cluster, not_in_cluster],
                                            [out_cluster, not_out_cluster]])

                if contingency_table.min() < 5:
                    test_name = 'fisher'
                    oddsratio, p_val = stats.fisher_exact(contingency_table)
                else:
                    test_name = 'chi2'
                    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                    oddsratio = (in_cluster * not_out_cluster) / max((out_cluster * not_in_cluster), 1)

                effect_size = np.log2(oddsratio) if oddsratio > 0 else 0  # Log2 odds ratio

                results.append([cluster, var, val, p_val, effect_size, test_name, 'log2_odds'])
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=['kmeans_7', 'var_name', 'var_value', 'p_val', 'effect_size', 'test_type', 'effect_type'])

    # Apply Benjamini-Hochberg FDR correction
    results_df['p_adj'] = multipletests(results_df['p_val'], method=method)[1]

    # compute absolute value of effect size
    results_df['abs_effect_size'] = np.abs(results_df['effect_size'])

    # compute -log10(p_adj)
    results_df['-log10(p_adj)'] = -np.log10(results_df['p_adj'])

    # Sort results by adjusted p-value for easier interpretation
    results_df = results_df.sort_values(by=['-log10(p_adj)', 'abs_effect_size'], ascending=False)

    return results_df