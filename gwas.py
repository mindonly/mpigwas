#!/usr/bin/env python3

import pandas as pd
import numpy as np

from math import sqrt


def tstat(sample1, sample2):
    n1 = sample1.size
    std1 = np.nanstd(sample1, ddof=1)
    mean1 = np.nanmean(sample1)

    n2 = sample2.size
    std2 = np.nanstd(sample2, ddof=1)
    mean2 = np.nanmean(sample2)

    return (mean1 - mean2) / (sqrt((std1 * std1 / n1) + (std2 * std2 / n2)))


def dscore(tstat, sample1, sample2):
    dset = np.concatenate([sample1, sample2])
    dset_std = np.std(dset)
    dset_mean = np.mean(dset)

    return abs(tstat - dset_mean) / dset_std


def example():
    dataset = "ex1.csv"
    df = pd.read_csv(dataset, delimiter=',')
    df = df.drop('Gene', 1)     # drop 'Gene' column
    tuples = [tuple(x) for x in df.values]

    print("gene\tT-statistic")
    print("-------------------")
    for i in range(0, len(tuples)):
        smp1 = np.array(tuples[i][1:4])
        print("non-blanks in sample1: ", np.count_nonzero(~np.isnan(smp1)))
        smp2 = np.array(tuples[i][4:7])
        print("non-blanks in sample2: ", np.count_nonzero(~np.isnan(smp2)))
        t = tstat(smp1, smp2)
        print(tuples[i][0], "\t", format(t, '.2f'))


def nci():
    dataset = "NCI-60.csv"
    df = pd.read_csv(dataset, delimiter=',')
    tuples = [tuple(x) for x in df.values]

    print("gene\tT-statistic")
    print("-------------------")
    # for i in range(0, len(tuples)):
    for i in range(0, 7):

        # raw observation
        raw_obs = np.array(tuples[i])

        # count non-blanks
        ren_set = np.array(tuples[i][1:9])
        # ren_non_bl = np.count_nonzero(~np.isnan(ren_set))
        con_set = np.array(tuples[i][9:])
        # con_non_bl = np.count_nonzero(~np.isnan(con_set))

        # base observation, reduced samples
        smp1_base = ren_set[~np.isnan(ren_set)]
        smp2_base = con_set[~np.isnan(con_set)]
        t_base = tstat(smp1_base, smp2_base)

        # construct reduced full observation
        red_full_obs = np.concatenate([smp1_base, smp2_base])

        print(tuples[i][0], "\t", format(t_base, '.2f'))


def main():
    # example()
    nci()


if __name__ == "__main__":
    main()
