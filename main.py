import argparse
import os
import sys
import time
from tqdm import tqdm
from genome_readers import reverse_complement, read_unitigs, read_genome
import numpy as np
import pandas as pd
from multiprocessing import Pool
from numpy.linalg import solve
import edlib
from run_cuttlefish import run_cuttlefish

def compute_S_D_I_N(u1, unitig_set_mutd, k):
    num_kmers_single_subst, num_kmers_single_delt, num_kmers_no_mutation = 0, 0, 0
    num_kmers_single_insertion = 0

    for u2 in unitig_set_mutd:
        alignment, distance, st1, st2 = None, 9999999999, None, None
        
        r1 = edlib.align(u1, u2, mode = "HW", task = "path")
        r2 = edlib.align(u2, u1, mode = "HW", task = "path")
        
        u3 = reverse_complement(u1)
        r3 = edlib.align(u3, u2, mode = "HW", task = "path")
        r4 = edlib.align(u2, u3, mode = "HW", task = "path")
        
        for i, r in enumerate([r1, r2, r3, r4]):
            if r['editDistance'] < distance:
                alignment, distance = r, r['editDistance']
                if i == 0:
                    st1, st2 = u1, u2
                    flip = False
                elif i == 1:
                    st1, st2 = u2, u1
                    flip = True
                elif i == 2:
                    st1, st2 = u3, u2
                    flip = False
                else:
                    st1, st2 = u2, u3
                    flip = True
        
        nice = edlib.getNiceAlignment(alignment, st1, st2)
        seqA, seqB = nice['query_aligned'], nice['target_aligned']
        assert len(seqA) == len(seqB)
        
        if flip:
            seqB, seqA = seqA, seqB
            
        alphabet = set('ACGT')
        num_chars = len(seqA)
        in_numbers = [0 for i in range(num_chars)]
        for i in range(num_chars):
            if seqA[i] != seqB[i]:
                if seqA[i] in alphabet and seqB[i] in alphabet:
                    in_numbers[i] = 1
                else:
                    in_numbers[i] = 2

        for i in range(num_chars-k+1):
            if sum(in_numbers[i:i+k]) == 1:
                num_kmers_single_subst += 1

        in_numbers = [0 for i in range(num_chars)]
        for i in range(num_chars):
            if seqB[i] == '-' and seqA[i] in alphabet:
                in_numbers[i] = 1
            elif seqA[i] != seqB[i]:
                in_numbers[i] = 2

        for i in range(num_chars-k+1):
            if sum(in_numbers[i:i+k]) == 1:
                num_kmers_single_delt += 1
            if sum(in_numbers[i:i+k]) == 0:
                num_kmers_no_mutation += 1
        
        in_numbers = [0 for i in range(num_chars)]
        for i in range(num_chars):
            if seqB[i] in alphabet and seqA[i] == '-':
                in_numbers[i] = 1
            elif seqA[i] != seqB[i]:
                in_numbers[i] = 2

        for i in range(num_chars-k+1):
            if sum(in_numbers[i:i+k]) == 1:
                num_kmers_single_insertion += 1
                    
                    
    return num_kmers_single_subst, num_kmers_single_delt, num_kmers_single_insertion, num_kmers_no_mutation


def compute_S_D_I_N_all(unitig_set_orig, unitig_set_mutd, k, num_threads = 64):
    # call compute_S_D_I_N using a multiprocessing pool
    # return the sum of all the values
    pool = Pool(num_threads)
    arg_list = [(u1, unitig_set_mutd, k) for u1 in unitig_set_orig]
    results = pool.starmap(compute_S_D_I_N, arg_list)
    pool.close()

    S, D, I, N = 0, 0, 0, 0
    for S_, D_, I_, N_ in results:
        S += S_
        D += D_
        I += I_
        N += N_

    return S, D, I, N


def estimate_rates(L, L2, N, D, fA, fA_mut, k):
    # use the equations to estimate the rates
    a1 = 1.0 * (L - fA) / 3.0 - fA
    b1 = - fA
    c1 = L/4.0
    d1 = fA_mut - fA

    a2 = 0
    b2 = -1
    c2 = 1
    d2 = 1.0*L2/L - 1

    a3 = 1
    b3 = 1.0 * N * k / D + 1
    c3 = 0
    d3 = 1.0

    A = np.array([[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]])
    b = np.array([d1, d2, d3])
    x = solve(A, b)

    subst_rate, del_rate, ins_rate = x

    return subst_rate, del_rate, ins_rate


def split_unitigs(unitigs, k):
    return_list = []
    for u in unitigs:
        if len(u) < 6000:
            return_list.append(u)
        else:
            num_splits = len(u) / 5000
            # round up
            num_splits = int(num_splits) + 1
            for i in range(num_splits):
                start = i * 5000
                end = min((i+1) * 5000, len(u))
                if start >= end:
                    break
                return_list.append(u[start:end])
                start = end - (k-1)
                end = min(end + k, len(u))
                if start >= end:
                    break
                return_list.append(u[start:end])
    return return_list


def compute_mutation_rates(genome_filename1, genome_filename2, k, num_threads = 255):
    orig_string = read_genome(genome_filename1)
    mutated_string = read_genome(genome_filename2)
    
    L = len(orig_string)
    L2 = len(mutated_string)
    
    fA = orig_string.count('A')
    fA_mut = mutated_string.count('A')
    
    genome1_cuttlefish_prefix = genome_filename1+"_unitigs"
    genome1_unitigs_filename = genome1_cuttlefish_prefix + ".fa"
    genome2_cuttlefish_prefix = genome_filename2+"_unitigs"
    genome2_unitigs_filename = genome2_cuttlefish_prefix + ".fa"
    
    run_cuttlefish(genome_filename1, k, 64, genome1_cuttlefish_prefix)
    run_cuttlefish(genome_filename2, k, 64, genome2_cuttlefish_prefix)
    
    assert os.path.exists(genome1_unitigs_filename), f"Mutated unitigs file {genome1_unitigs_filename} not found"
    assert os.path.exists(genome2_unitigs_filename), f"Original unitigs file {genome2_unitigs_filename} not found"
    
    # read two sets of unitigs
    unitig_set_orig = read_unitigs(genome1_unitigs_filename)
    unitig_set_mutd = read_unitigs(genome2_unitigs_filename)
    
    # split unitigs into smaller unitigs
    unitig_set_orig = split_unitigs(unitig_set_orig, k)
    unitig_set_mutd = split_unitigs(unitig_set_mutd, k)
    
    # compute S, D, I, N
    S, D, I, N = compute_S_D_I_N_all(unitig_set_orig, unitig_set_mutd, k, num_threads)
    
    # compute the rates
    subst_rate, del_rate, ins_rate = estimate_rates(L, L2, N, D, fA, fA_mut, k)
    
    return subst_rate, del_rate, ins_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mutation rates")
    parser.add_argument("genome_filename1", type=str, help="Genome filename")
    parser.add_argument("genome_filename2", type=str, help="Genome filename")
    parser.add_argument("k", type=int, help="k-mer size")
    parser.add_argument("--num_threads", type=int, default=255, help="Number of threads to use")
    args = parser.parse_args()

    subst_rate, del_rate, ins_rate = compute_mutation_rates(args.genome_filename1, args.genome_filename2, args.k, args.num_threads)
    print(f"Substitution rate: {subst_rate}")
    print(f"Deletion rate: {del_rate}")
    print(f"Insertion rate: {ins_rate}")