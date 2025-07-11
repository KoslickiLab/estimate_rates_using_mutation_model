from Bio import SeqIO

def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join([complement[base] for base in reversed(seq)])

def read_genome(genome_file):
    """
    Reads a genome file and returns the genome as a string
    """
    genome = ""
    for record in SeqIO.parse(genome_file, "fasta"):
        genome += str(record.seq)
    return clean_genome_string(genome)

def clean_genome_string(genome_string):
    """
    Removes all non-alphabet characters from a genome string
    """
    alphabet = set('ACGT')
    genome_string = genome_string.upper() # in case there are lower cases in the given string.
    return ''.join(filter(alphabet.__contains__, genome_string))

def get_genome_length(genome_filename):
    """
    Returns the length of the genome in the genome file
    """
    genome = read_genome(genome_filename)
    return len(clean_genome_string(genome))

def get_kmers(genome_string, k):
    """
    Returns a list of all k-mers in a genome string
    """
    kmers = []
    for i in range(len(genome_string)-k+1):
        kmers.append(genome_string[i:i+k])
    return kmers

def read_unitigs(unitigs_file):
    unitigs = set()
    with open(unitigs_file) as f:
        for line in f:
            if line[0] == '>':
                continue
            else:
                unitigs.add(line.strip())
    return list(unitigs)