# DeepWheat Introduction
DeepWheat, a two-part deep learning framework—DeepEXP and DeepEPI—that enables precise, tissue-specific prediction of gene expression. DeepEXP leverages multi-omic inputs to achieve high accuracy, especially for tissue-specific genes, while DeepEPI predicts epigenomic features directly from DNA sequence, facilitating model transfer across varieties.  DeepWheat further identifies regulatory variants with strong expression effects, supporting targeted CRE editing and providing a versatile toolset for functional genomics and trait improvement in crops.
![image](https://github.com/user-attachments/assets/f37ae380-d3a9-40d8-a880-532703c0ceb5)
# DeepEpi Usage Guide
## Install basenji-3.9 (https://github.com/calico/basenji)  
You can find usage guide in /DeepEpi 

# EeepExp  Install and Usage Guide
wget https://zenodo.org/records/15765929/files/DeepExp_env.tar.gz   
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

