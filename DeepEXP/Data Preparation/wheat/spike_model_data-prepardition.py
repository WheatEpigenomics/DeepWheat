import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
sequence_file_path = './final-target-seq.fa'
epi_file_path = './spike_all_epi.csv'
# load and one-hot encoding sequence data 
def txt_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    return lines
sequences = txt_to_list(sequence_file_path)

def data_encode(seqs):
    dna = np.zeros((len(seqs), len(seqs[0]), 4), dtype='float16')
    for i in tqdm(range(len(seqs))):
        for j in range(len(seqs[i])):
            if seqs[i][j] == 'A':
                dna[i][j][0] = 1
            elif seqs[i][j] == 'C':
                dna[i][j][1] = 1
            elif seqs[i][j] == 'G':
                dna[i][j][2] = 1
            elif seqs[i][j] == 'T':
                dna[i][j][3] = 1
           
    return dna
encoded_gene_sequences = data_encode(sequences)
#  To save one-hot encoded data into a HDF5 file
with h5py.File('spike_sequence.h5', 'w') as f:
    f.create_dataset('dataset_2', data=encoded_gene_sequences)

# load epigenomic data 
df = pd.read_csv(epi_file_path)
epigenome = MinMaxScaler().fit_transform(df.values[:, :100000])
# To save deal epigenomic data into other HDF5 file
with h5py.File('spike_epi.h5', 'w') as f:
    f.create_dataset('dataset_1', data=epigenome)
# load data 
with h5py.File('spike_sequence.h5', 'r') as hf:
    sequence_input_data = hf['dataset_2'][:]
with h5py.File('spike_epi.h5', 'r') as hf:
    epigenetics_input_data = hf['dataset_1'][:]

# Check the dimensions of the epigenomic data and sequence data
print(f"Epigenetics data shape: {epigenetics_input_data.shape}")
print(f"Sequence data shape: {sequence_input_data.shape}")

