#!/bin/bash 
#PBS -q regular 
#PBS -N get-seq
#PBS -l walltime=100000:00:00
#PBS -l nodes=1:ppn=2
conda activate tensorflow2.8
python predict-have-true_gene_expression.py
python predict-not-true_gene_expression.py
