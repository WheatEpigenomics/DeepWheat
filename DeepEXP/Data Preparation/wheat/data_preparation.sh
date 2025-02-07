# 01 sequence data pre
#!/bin/bash 
#PBS -q regular 
#PBS -N get-seq
#PBS -l walltime=100000:00:00
#PBS -l nodes=1:ppn=2
# Obtain the sequences of the upstream and downstream 3K of the transcription start site (TSS) and the upstream and downstream 2K of the transcription termination site (TTS) for the positive-strand genes.
bedtools getfasta -fi ./data/AK58_genome.fa -bed pos-gene_3K-TSS-3K.bed -fo pos-gene-3K-TSS-3K.fa -name
bedtools getfasta -fi  ./data/AK58_genome.fa -bed pos-gene_2K-TTS-2K.bed  -fo pos-gene-2K-TTS-2K.fa -name
paste pos-gene-3K-TSS-3K.fa pos-gene-2K-TTS-2K.fa | awk '{print $1 $2}' | grep -v ">"  > pos-gene_sequence.fa
# Obtain the sequences of the upstream and downstream 3K of the transcription start site (TSS) and the upstream and downstream 2K of the transcription termination site (TTS) for the negative-strand genes.
bedtools getfasta -fi  ./data/AK58_genome.fa -bed nega-gene_3K-TSS-3K.bed -fo nega-gene-3K-TSS-3K.fa -name
bedtools getfasta -fi  ./data/AK58_genome.fa -bed nega-gene_2K-TTS-2K.bed  -fo nega-gene-2K-TTS-2K.fa -name
paste nega-gene-2K-TTS-2K.fa  nega-gene-3K-TSS-3K.fa | awk '{print $1 $2}' | grep -v ">"  > nega-gene-target_sequence.fa
awk '{print | "rev"}' nega-gene-target_sequence.fa > T-nega-gene-target_sequence.fa
cat pos-gene-target_sequence.fa T-nega-gene-target_sequence.fa > final-target-seq.fa
# 02  epi data pre 
#A01 Obtain the normalized  bedGraph file of epigenomic
#A02 For each target gene, assign the normalized epigenomic signal values to the regions 3K upstream and 2K downstream of the TSS and 2K upstream and 2K downstream of the TTS, and convert the file to a CSV file
# positive-strand genes  
python S1_find_target_site_value.py spike-atac.norm.bedGraph pos-gene_3K-TSS-3K.bed pos-gene-TSS_3K-spike_ATAC_value-1
python S1_find_target_site_value.py spike-atac.norm.bedGraph pos-gene_2K-TTS-2K.bed pos-gene-TTS_2K-spike_ATAC_value-1
cat pos-gene-TSS_3K-spike_ATAC_value-1 | cut -f 4 >pos-gene-TSS_3K-spike_ATAC_value-2
python S2_transpose-tss.py pos-gene-TSS_3K-spike_ATAC_value-2 pos-gene-TSS_3K-YP_ATAC_value-final
cat pos-gene-TTS_2K-spike_ATAC_value-1 | cut -f 4 > pos-gene-TTS_2K-spike_ATAC_value-2
python S3_transpose-tts.py pos-gene-TTS_2K-spike_ATAC_value-2 pos-gene-TTS_2K-YP_ATAC_value-final
paste pos-gene-TSS_3K-spike_ATAC_vaue-final pos-gene-TTS_2K-spike_ATAC_value-final > pos-gene-spike_ATAC.value 
# negative-strand genes 
python S1_find_target_site_value.py spike-atac.norm.bedGraph nega-gene_3K-TSS-3K.bed nega-gene-TSS_3K-spike_ATAC_value-1
python S1_find_target_site_value.py spike-atac.norm.bedGraph nega-gene_2K-TTS-2K.bed nega-gene-TTS_2K-spike_ATAC_value-1
cat  nega-gene-TSS_3K-spike_ATAC_value-1 | cut -f 4 > nega-gene-TSS_3K-spike_ATAC_value-2
python S2_transpose-tss.py nega-gene-TSS_3K-spike_ATAC_value-2 nega-gene-TSS_3K-spike_ATAC_value-final
cat fu-TTS_2K-spike_ATAC_value-1 | cut -f 4 > nega-gene-TTS_2K-spike_ATAC_value-2
python S3_transpose-tts.py nega-gene-TTS_2K-spike_ATAC_value-2 nega-gene-TTS_2K-spike_ATAC_value-final
paste nega-gene-TTS_2K-spike_ATAC_value-final nega-gene-TSS_3K-spike_ATAC_value-final > nega-gene-spike_ATAC.final
awk '{for(i=NF;i>0;i--) printf "%s%s", $i, (i>1? "\t" : ORS)}' nega-gene-spike_ATAC.final > T-nega-gene-spike_ATAC.value
# A03 obtain csv file of epigenomic
paste pos-gene-spike_ATAC.value  pos-gene-spike_K27AC.value  pos-gene-spike_K27.value  pos-gene-spike_K36.value  pos-gene-spike_K4.value > pos-gene-spike_all_epi_data
paste T-nega-gene-spike_ATAC.value T-nega-gene-spike_K27AC.value T-nega-gene-spike_K27.value T-nega-gene-spike_K36.value T-nega-gene-spike_K4.value > T-nega-gene-spike_all_epi_data
cat pos-gene-spike_all_epi_data  T-nega-gene-spike_all_epi_data | awk 'BEGIN {printf "S1"; for (i=2; i<=50000; i++) printf "\tS%d", i; print ""} {print}' | YP_all_epi_data-1 | tr "\t" ","  > YP_all_epi.csv
rm *final *value-1  *value-2  *value
# 03 Process the final sequences and epigenomic data for model training
python spike_model_data-prepardition.py 



