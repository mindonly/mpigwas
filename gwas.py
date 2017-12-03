#!/usr/bin/env python3

import pandas as pd
import numpy as np
import timeit

from math import sqrt


# return a t-statistic for two samples
def tstat(sample1, sample2):
    n1 = sample1.size
    std1 = np.std(sample1, ddof=1)
    mean1 = np.mean(sample1)

    n2 = sample2.size
    std2 = np.std(sample2, ddof=1)
    mean2 = np.mean(sample2)

    return (mean1 - mean2) / (sqrt((std1 * std1 / n1) + (std2 * std2 / n2)))


# return a D-score given a T-statistic and a distribution
def dscore(tstat, dist):
    dist_std = np.std(dist)
    dist_mean = np.mean(dist)

    return abs(tstat - dist_mean) / dist_std


# create a random T-statistic distribution from a gene observation
# remain fixed at 1000 permutations
def create_distribution(obs, k):
    n = 1000
    d = np.zeros(n)
    for i in range(0, n):
        perm_obs = np.random.permutation(obs)
        smp1 = perm_obs[0:k]
        smp2 = perm_obs[k:]
        d[i] = tstat(smp1, smp2)

    return d


# reproduce the small assignment example
def example():
    dataset = "ex1.csv"
    df = pd.read_csv(dataset, delimiter=',')
    df = df.drop('Gene', 1)     # drop 'Gene' column
    tuples = [tuple(x) for x in df.values]

    print("\nAssignment Example:")
    print("-------------------")
    print("gene\tT-statistic")
    print("-------------------")
    for i in range(0, len(tuples)):
        smp1 = np.array(tuples[i][1:4])
        smp2 = np.array(tuples[i][4:7])
        t = tstat(smp1, smp2)
        print(tuples[i][0], "\t", format(t, '.2f'))


# process NCI-60 gene expression data
def nci():
    dataset = "NCI-60.csv"
    df = pd.read_csv(dataset, delimiter=',')
    tuples = [tuple(x) for x in df.values]
    tmp_df = pd.DataFrame(columns=['index', 'gene', 'T-stat', 'D-score'])

    print("processing", dataset, "...")
    for i in range(0, len(tuples)):
        gene_name = tuples[i][0]

        # split the observation, renal cancer and control
        ren_set = np.array(tuples[i][1:9])
        con_set = np.array(tuples[i][9:])

        # reference observation, reduced samples (no blanks, NaNs)
        smp1_ref = ren_set[~np.isnan(ren_set)]
        smp2_ref = con_set[~np.isnan(con_set)]

        # construct reduced full observation
        reduced_obs = np.concatenate([smp1_ref, smp2_ref])

        # generate random t-stat distribution for a gene
        # based on number of renal cancer patients for that gene
        ren_count = len(smp1_ref)
        obs_dist = create_distribution(reduced_obs, ren_count)
        t_ref = tstat(smp1_ref, smp2_ref)
        d_score = dscore(t_ref, obs_dist)

        row = np.array([i+2, gene_name, format(t_ref, '.3f'), format(d_score, '.3f')])
        tmp_df.loc[i] = row

    results_df = tmp_df.sort_values(['D-score'], ascending=False)
    results_df.to_csv("results.csv", index=False)


def main():
    # example()

    nci_start_time = timeit.default_timer()
    nci()
    print("elapsed time:", format(timeit.default_timer() - nci_start_time, '.2f'), "s.")


if __name__ == "__main__":
    main()
