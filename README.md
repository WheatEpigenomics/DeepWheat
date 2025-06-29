# DeepWheat Introduction
DeepWheat, a two-part deep learning framework—DeepEXP and DeepEPI—that enables precise, tissue-specific prediction of gene expression. DeepEXP leverages multi-omic inputs to achieve high accuracy, especially for tissue-specific genes, while DeepEPI predicts epigenomic features directly from DNA sequence, facilitating model transfer across varieties.  DeepWheat further identifies regulatory variants with strong expression effects, supporting targeted CRE editing and providing a versatile toolset for functional genomics and trait improvement in crops.
![image](https://github.com/user-attachments/assets/f37ae380-d3a9-40d8-a880-532703c0ceb5)
# DeepEpi Usage Guide
## Install basenji-3.9 (https://github.com/calico/basenji)
## data pre and model train 
bam_cov.py -a  sample.sort.bam sample.a.cov.bw
basenji_data.py -g ./data/gap_out_10.bed -l 131072 --local -o ./ATAC_data_131k -p 24 -t .2 -v .2 -w 128 ./AK58.fa ./data/atac.target.txt 
basenji_data.py -g ./data/gap_out_10.bed -l 131072 --local -o ./EPI_data_131k -p 24 -t .2 -v .2 -w 128 ./AK58.fa ./data/epi.target.txt 
basenji_data.py -g ./data/gap_out_10.bed -l 131072 --local -o ./RNA_data_131k -p 24 -t .2 -v .2 -w 128 ./AK58.fa ./data/rna.target.txt
basenji_train.py -o ./models/ATAC_train_models ./data/atac_params-NM.json ./ATAC_data_131k
basenji_train.py -o ./models/EPI_train_models  ./data/epi_params-NM.json ./EPI_data_131k
basenji_train.py -o ./models/RNA_train_models ./data/RNA-params-NM.json ./RNA_data_131
model test
basenji_test.py --ai 0,1,2,3,4,5,6 --bi 0,1,2,3,4,5 --peak  -o ./ATAC_test --rc ./data/atac_params-NM.json ./model/ATAC_train_models/model_best.h5 ./ATAC_data_131k
basenji_test.py --ai 0,1,2,3,4,5,6 --bi 0,1,2,3,4,5 --peak  -o ./EPI_test --rc ./data/epi_params-NM.json ./model/EPI_train_models/model_best.h5 ./EPI_data_131k
basenji_test.py --ai 0,1,2,3,4,5,6 --bi 0,1,2,3,4,5 --peak  -o ./RNA_test --rc ./data/RNA-params-NM.json ./model/RNA_train_models/model_best.h5 ./RNA_data_131k
 
atac predict
basenji_predict_bed.py -b 0,1,2,3,4,5  -f ./AK58.fa -g ./data/AK58.genome.sizes -o AK58-atac_predict_out --rc -t ./data/ATAC.target.txt ./data/atac_params-NM.json ./model/atac_train_models/model_best.h5 ./data/predict.bed
#epigenome predict
basenji_predict_bed.py -b 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23  -f ./AK58.fa -g ../data/AK58.genome.sizes -o AK58-epi_predict_out --rc -t  ./data/atac.target.txt ./data/epi_params-NM.json ./data/EPI_train_models/model_best.h5 ./data/predict.bed
gene expression predict
basenji_predict_bed.py -b 0,1,2,3,4,5  - -f./AK58.fa -g ./data/AK58.genome.sizes   -o target-gene_RNA_predict_out --rc -t ./data/rna.target.txt ./data/rna-params-NM.json ./model/rna_train_models/model_best.h5  ./data/predict_gene.bed
SNP effect  caculate 
basenji_sad.py -f ./data/AK58.fa -o sad_out --rc -t ./atac.target.txt --shift "1,0,-1" --ti "0,1" ./data/atac_params-NM.json ./model/ATAC_train_models/model_best.h5 ./data/predict.vcf
basenji_sad_table.py ./sad_out/sad.h5
Saturation mutagenesis 
basenji_sat_bed.py -f ./AK58.fa -l 200 -d 100 -u 100 -o sat-mut --plots --rc -t ./data/atac.target.txt ./data/atac_params-NM.json ./mdeol/ATAC_train_models/model_best.h5 ./data/predict.vcf
basenji_sat_plot.py ./sat-mut/scores.h5
# EeepExp  Install and Usage Guide
https://zenodo.org/records/15765929/files/DeepExp_env.tar.gz
mkdir -p ~/apps/DeepExp_env
tar -xzf DeepExp_env.tar.gz -C ~/apps/DeepExp_env
cd ~/apps/DeepExp_env
./bin/conda-unpack
source ~/apps/DeepExp_env/bin/activate
cd project/script
python get_seq.py -h                 
python model_train.py -h
python predict_variant_effect.py -h


# Environment and package Environment
You can find in requirements.txt
# Questions
If you have any questions, requests, or comments, we kindly invite you to contact us at mzgcckk@163.com, luzefu@caas.cn.

