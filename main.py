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

def wrapper(args):
    return compute_S_D_I_N(*args)

def compute_S_D_I_N_all(unitig_set_orig, unitig_set_mutd, k, num_threads=64):
    arg_list = [(u1, unitig_set_mutd, k) for u1 in unitig_set_orig]
    
    S, D, I, N = 0, 0, 0, 0

    with Pool(num_threads) as pool:
        for result in tqdm(pool.imap(wrapper, arg_list), total=len(arg_list), desc="Processing"):
            S_, D_, I_, N_ = result
            S += S_
            D += D_
            I += I_
            N += N_

    return S, D, I, N


def estimate_rates_polynomial(L, L2, S, D, I, k):
    K1 = L - k + 1
    K2 = L2 - k + 1
    
    S_norm = 1.0 * S / (K1 * k)
    D_norm = 1.0 * D / (K1 * k)
    I_norm = 1.0 * I / (K1 * k - K1)
    
    coeffs = [0 for i in range(k+1)]
    coeffs[0] = (S_norm + D_norm + I_norm) * D_norm**k
    coeffs[-1] = 1
    coeffs[-2] = -D_norm
    
    roots = np.polynomial.polynomial.polyroots(coeffs)
    
    p_d_ests = (D_norm - roots)/(S_norm + D_norm + I_norm)
    p_d_ests = [np.real(p_d_est) for p_d_est in p_d_ests if not np.iscomplex(p_d_est)]
    p_d_ests.sort()
    
    # drop the negative roots
    p_d_ests = [p_d_est for p_d_est in p_d_ests if p_d_est >= 0]
    
    if len(p_d_ests) == 0:
        return 0, 0, 0
    
    p_d_est = p_d_ests[0]
    d_est = (D_norm - (S_norm + D_norm) * p_d_est)/(D_norm - (S_norm + D_norm + I_norm) * p_d_est) - 1.0
    p_s_est = (S_norm * p_d_est)/(D_norm)
    
    return p_s_est, p_d_est, d_est


def estimate_rates_linear(L, L2, N, D, fA, fA_mut, k):
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
    
    p_s = 3 * ( N*k*(L2-4*fA_mut-L+4*fA) + D*(L2-4*fA_mut) ) / ( (L - 4*fA) * (3 - 4*N*k - 4*D))
    print(p_s, subst_rate)

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
    fC = orig_string.count('C')
    fG = orig_string.count('G')
    fT = orig_string.count('T')
    fA_mut = mutated_string.count('A')
    fC_mut = mutated_string.count('C')
    fG_mut = mutated_string.count('G')
    fT_mut = mutated_string.count('T')
    
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
    
    # DEBUG: print L, L2, S, D, I, N, fA, fA_mut, k
    print(f"DBG: L: {L}, L2: {L2}, S: {S}, D: {D}, I: {I}, N: {N}, fA: {fA}, fA_mut: {fA_mut}, k: {k}")
    # DEBUG: show fA, fC, fG, fT
    print(f"DBG: fA: {fA}, fC: {fC}, fG: {fG}, fT: {fT}")
    
    largest = max(fA, fC, fG, fT)
    if largest==fA:
        pass
    elif largest==fC:
        fA, fA_mut = fC, fC_mut
    elif largest==fG:
        fA, fA_mut = fG, fG_mut
    elif largest==fT:
        fA, fA_mut = fT, fT_mut
    else:
        raise ValueError("Invalid largest value")
    
    # compute the rates
    subst_rate_lin, del_rate_lin, ins_rate_lin = estimate_rates_linear(L, L2, N, D, fA, fA_mut, k)
    subst_rate_poly, del_rate_poly, ins_rate_poly = estimate_rates_polynomial(L, L2, S, D, I, k)
    
    return subst_rate_lin, del_rate_lin, ins_rate_lin, subst_rate_poly, del_rate_poly, ins_rate_poly


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mutation rates")
    parser.add_argument("genome_filename1", type=str, help="Genome filename")
    parser.add_argument("genome_filename2", type=str, help="Genome filename")
    parser.add_argument("k", type=int, help="k-mer size")
    parser.add_argument("--num_threads", type=int, default=255, help="Number of threads to use")
    args = parser.parse_args()

    subst_rate_lin, del_rate_lin, ins_rate_lin, subst_rate_poly, del_rate_poly, ins_rate_poly = compute_mutation_rates(args.genome_filename1, args.genome_filename2, args.k, args.num_threads)
    print(f"Linear solution: Substitution rate: {subst_rate_lin}, Deletion rate: {del_rate_lin}, Insertion rate: {ins_rate_lin}")
    print(f"Polynomial solution: Substitution rate: {subst_rate_poly}, Deletion rate: {del_rate_poly}, Insertion rate: {ins_rate_poly}")