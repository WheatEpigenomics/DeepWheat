# DeepWheat Introduction
DeepWheat, a two-part deep learning framework—DeepEXP and DeepEPI—that enables precise, tissue-specific prediction of gene expression. DeepEXP leverages multi-omic inputs to achieve high accuracy, especially for tissue-specific genes, while DeepEPI predicts epigenomic features directly from DNA sequence, facilitating model transfer across varieties.  DeepWheat further identifies regulatory variants with strong expression effects, supporting targeted CRE editing and providing a versatile toolset for functional genomics and trait improvement in crops.
![image](https://github.com/user-attachments/assets/f37ae380-d3a9-40d8-a880-532703c0ceb5)
# DeepEpi  Install and Usage Guide  
DeepEXP was developed based on Basenji-3.9. You can view and install the dependencies via environment-deepepi.yml for conda. The codebase is compatible with the latest TensorFlow 2  
conda env create -f environment-deepepi.yml  
conda install tensorflow (or tensorflow-gpu)  
python setup.py develop --no-deps

# DeepExp  Install and Usage Guide  
DeepEXP was developed using Python 3.8 and various scientific computing dependencies, which can be viewed and installed via conda. The codebase is compatible with TensorFlow 2.8.  
wget https://zenodo.org/records/15765929/files/DeepExp_env.tar.gz   
tar -xzf DeepExp_env.tar.gz -C ~/apps/DeepExp_env  
cd ~/apps/DeepExp_env  
./bin/conda-unpack  
source ~/apps/DeepExp_env/bin/activate  
cd project/script  
python get_seq.py -h  
python get_epi.py -h  
python model_train.py -h   
python predict_gene_expression-IG_analysis.py -h   
python predict_variant_effect.py -h


# Environment and package Environment
You can find in requirements.txt and environment-deepepi.yml
# Questions
If you have any questions, requests, or comments, we kindly invite you to contact us at mzgcckk@163.com, luzefu@caas.cn.

