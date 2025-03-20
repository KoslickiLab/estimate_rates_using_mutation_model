import os
import subprocess

def run_cuttlefish(genome_filename, k, num_threads, outoput_prefix):
    #rm random_mutated.fasta_unitigs*
    #cuttlefish build -s random_mutated.fasta -k 21 -t 128 -o random_mutated.fasta_unitigs -w . --ref
    
    #cmd = f"rm {outoput_prefix}*"
    #os.system(cmd)

    cmd = f"cuttlefish build -s {genome_filename} -k {k} -t {num_threads} -o {outoput_prefix} -w . --ref"
    
    with open(os.devnull, 'wb') as devnull:
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=subprocess.STDOUT)