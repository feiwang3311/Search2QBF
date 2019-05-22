# Search2QBF

This repo depends on tensorflow_ranking (https://github.com/tensorflow/ranking), which should be installed following their readme file.

## Neural2QBF
### to generate random QBF instances
python3 randQBF.py --n_quantifiers 2 -a 3 -a 3 -a 8 -a 8 --n_clauses 50 --n_problems 10 --target_dir 3_3_8_8_50_10

### to generate paired random QBF instances (differ by one lit in formula, but differ completely in sat/unsat)
python3 randQBFinc.py --n_quantifiers 2 -a 2 -a 3 -a 2 -a 3 --n_clauses 10 --n_pairs 10 --target_dir 2_3_2_3_10_10 

### to transform dimacs to pickle dump (need to find optimal max_node_per_batch to maximize the efficiency of GPU memory)
python3 dimacs_to_data.py --dimacs_dir /u/data/u99/wang603/QBF/train10 --out_dir ./train10/ --max_nodes_per_batch 5000 --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10

### to train
python3 train.py --train_dir ./train10/ --run_id 0 


## Search2QBF
### use the paired random QBF instances generated for Neural2QBF (only use the unsat instances)
### transform 20 files to problem 
python3 dimacs_to_data.py --dimacs_dir /homes/wang603/QBF/train10_unsat/ --out_dir ./train10_unsat/ --max_dimacs 20 --max_nodes_per_batch 5000 --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10

### transform all 1000 files to problem
python3 dimacs_to_data.py --dimacs_dir /homes/wang603/QBF/train10_unsat/ --out_dir ./train10_unsat/ --max_nodes_per_batch 5000 --n_quantifiers 2 -a 2 -a 3 -a 8 -a 10

### before running, activate tensorflow_ranking environment
source ~/QBF/tfr/bin/activate

### train with 1000 problems
python3 train.py --train_dir ./train10_unsat/ --run_id 3
