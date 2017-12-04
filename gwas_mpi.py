#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os, time, timeit

from mpi4py import MPI
from math import sqrt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()              # number of nodes
hostname = MPI.Get_processor_name() # hostname
MASTER = 0


# return T-statistic for two samples
def tstat(sample1, sample2):
    n1 = sample1.size
    std1 = np.std(sample1, ddof=1)
    mean1 = np.mean(sample1)

    n2 = sample2.size
    std2 = np.std(sample2, ddof=1)
    mean2 = np.mean(sample2)

    return (mean1 - mean2) / (sqrt((std1 * std1 / n1) + (std2 * std2 / n2)))


# return D-score given T-statistic and distribution
def dscore(tstat, dist):
    dist_std = np.std(dist)
    dist_mean = np.mean(dist)

    return abs(tstat - dist_mean) / dist_std


# create random T-statistic distribution from gene row observation
# fixed at 1000 permutations
def create_distribution(obs, k):
    n = 1000
    d = np.zeros(n)
    for i in range(0, n):
        perm_obs = np.random.permutation(obs)
        smp1 = perm_obs[0:k]
        smp2 = perm_obs[k:]
        d[i] = tstat(smp1, smp2)

    return d


# reproduce small assignment example
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


# process NCI-60 gene expression data sequentially
def seq_nci():
    # start timer
    start_time = timeit.default_timer()

    # prepare data
    dataset = "NCI-60.csv"
    df = pd.read_csv(dataset, delimiter=',')
    tuples = [tuple(x) for x in df.values]
    tmp_df = pd.DataFrame(columns=['index', 'gene', 'T-stat', 'D-score'])

    print("processing", dataset, "sequentially ...")
    for i in range(0, len(tuples)):
        gene_name = tuples[i][0]

        # split observation, renal cancer and control
        ren_set = np.array(tuples[i][1:9])
        con_set = np.array(tuples[i][9:])

        # reference observation, reduced samples (no blanks, NaNs)
        smp1_ref = ren_set[~np.isnan(ren_set)]
        smp2_ref = con_set[~np.isnan(con_set)]

        # construct reduced full observation
        reduced_obs = np.concatenate([smp1_ref, smp2_ref])

        # generate random t-stat distribution for gene
        # based on # of renal cancer patients for that gene
        ren_count = len(smp1_ref)
        obs_dist = create_distribution(reduced_obs, ren_count)
        t_ref = tstat(smp1_ref, smp2_ref)
        d_score = dscore(t_ref, obs_dist)

        # build row, dataframe append
        row = np.array([i+2, gene_name, format(t_ref, '.3f'), format(d_score, '.3f')])
        tmp_df.loc[i] = row

    # sort, write dataframe to csv
    results_file = "seq_results" + time.strftime("_%H%M%S") + ".csv"
    results_df = tmp_df.sort_values(['D-score'], ascending=False)
    results_df.to_csv(results_file, index=False)

    # stop timer
    end_time = timeit.default_timer()
    print("elapsed time (sequential:", format(end_time - start_time, '.2f'), "s")


# process NCI-60 gene expression data using MPI
def mpi_nci():
    if rank == MASTER:
        # start timer
        start_time = MPI.Wtime()

        # prepare data
        dataset = "NCI-60.csv"
        df = pd.read_csv(dataset, delimiter=',')
        tuples = [tuple(x) for x in df.values]

        # split, send the data
        data = np.array_split(tuples, size-1)
        for i in range(1, size):
            comm.send(data[i-1], dest=i)
       
        comm.Barrier()
        
        # setup receiving dataframe; receive processed dataframes
        tmp_df = pd.DataFrame(columns=['index', 'gene', 'T-stat', 'D-score'])
        for i in range(1, size):
            in_df = comm.recv(source=i)
            tmp_df = pd.concat([tmp_df, in_df])

        # sort, write results dataframe to csv
        results_file = "mpi_results" + time.strftime("_%H%M%S") + ".csv"
        results_df = tmp_df.sort_values(['D-score'], ascending=False)
        results_df.to_csv(results_file, index=False)
       
        # stop timer
        end_time = MPI.Wtime()
        #print("elapsed time (MPI):", format(end_time - start_time, '.2f'), "s")
        print(format(end_time - start_time, '.2f'))

    else:
        # setup working dataframe; receive dataset chunks
        tmp_df = pd.DataFrame(columns=['index', 'gene', 'T-stat', 'D-score'])
        data = comm.recv(source=MASTER)
        
        for i in range(0, len(data)):
            gene_name = data[i][0]

            # split observation, renal cancer and control
            ren_set = np.array(data[i][1:9], dtype=np.float64)
            con_set = np.array(data[i][9:], dtype=np.float64) 

            # reference observation, reduced samples (no blanks, NaNs)
            smp1_ref = ren_set[~np.isnan(ren_set)]
            smp2_ref = con_set[~np.isnan(con_set)]

            # construct reduced full observation
            reduced_obs = np.concatenate([smp1_ref, smp2_ref])

            # generate random t-stat distribution for gene
            # based on # of renal cancer patients for that gene
            ren_count = len(smp1_ref)
            obs_dist = create_distribution(reduced_obs, ren_count)
            t_ref = tstat(smp1_ref, smp2_ref)
            d_score = dscore(t_ref, obs_dist)
        
            # build rows, dataframe append
            row = np.array([i+2, gene_name, format(t_ref, '.3f'), format(d_score, '.3f')])
            tmp_df.loc[i] = row
            
        comm.Barrier()

        # send completed dataframe back to MASTER
        comm.send(tmp_df, dest=MASTER)


# get best genes from csv results output
# count arg specifies # of lines to pull from each csv file "head"
def best_genes(mode, count):
    workdir = "/home/sanchrob/cis677/Proj4/"
    df = pd.DataFrame(columns=['index', 'gene', 'T-stat', 'D-score'])

    for file in os.listdir(workdir):
        if mode == "mpi":
            if file.startswith("mpi_results") and file.endswith(".csv"):
                file_df = pd.read_csv(file, delimiter=',')
                file_df = file_df.head(count)
                df = pd.concat([df, file_df])
                best_df = df.sort_values(['gene', 'D-score'])
        elif mode == "seq":
            if file.startswith("seq_results") and file.endswith(".csv"):
                file_df = pd.read_csv(file, delimiter=',')
                file_df = file_df.head(count)
                df = pd.concat([df, file_df])
                best_df = df.sort_values(['gene', 'D-score'])
    
    print(mode, "counts:")
    print("-----------")
    print(best_df['gene'].value_counts())


def main():
    # example()
    # seq_nci()
    # mpi_nci()
    best_genes("seq", 15)
    best_genes("mpi", 15)


if __name__ == "__main__":
    main()
