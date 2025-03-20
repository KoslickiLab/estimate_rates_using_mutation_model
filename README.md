# Summary
This repository computes insertion, deletion, and substitution rates simultaneously between two input genomes under a non-simple mutation model. 

## Model description
To be added

## Estimators for the three mutation rates
To be added

## Workflow
The code in this repository does the following:

1. It first extracts k-mers from both input genome files
2. For each input, the program builds the unitigs from k-mers using cuttlefish. Let the set of unitigs built from the two genomes be $U_1$ and $U_2$
3. For a unitig $u$ in $U_1$, we align it to all unitigs in $U_2$. The alignment can be expensive for long unitigs. To make this feasible, we split long unitigs into smaller unitigs of length 5k
4. Next, for a unitig $u$ in $U_1$, we align it to all unitigs in $U_2$ using edlib. We get the best alignment using alignment score, and count the number of k-mers using single substitution, insertion, and deletion
5. These counts are used to solve our estimators for the three mutation rates

## Installation
```
conda create -n <environment_name> --file requirements.txt -c conda-forge -c bioconda
conda activate <environment_name>
```


## Usage
To be added