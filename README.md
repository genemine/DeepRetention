# DeepRetention
## Description
This is an implementation of the paper [DeepRetention](http://xxx) Zhen-Peng Wu, Jian-Tao Zheng, Jia-Shu Liu, Cui-Xiang Lin, Hong-Dong Li

Bibtex:
```sh
# todo
@article{Wu2021DeepRetention,
  title={DeepRetention: an deep learning approach for intron retention detection.},
  author={Zhen-Peng Wu, Jian-Tao Zheng, Jia-Shu Liu, Cui-Xiang Lin, Hong-Dong Li},
  year={2021},
}
```
## Requirements
* python >= 3.8
* gensim >= 3.8.3
* pandas >= 1.0.5
* numpy >= 1.18.5
* tensorflow >= 2.0
* samtools >= 1.7
* bedtools >= 2.29.2

## Install DeepRetention
```python
# python packages
pip install numpy==1.18.5 tensorflow==2.3.1 pandas==1.0.5 gensim==3.8.3

# bio tools: samtools 1.7 && bedtools 2.29.2
conda install samtools bedtools -c bioconda

# install from github
git clone https://github.com/genemine/DeepRetention.git
or
wget https://github.com/genemine/DeepRetention/archive/refs/heads/master.zip

### then download dataset.zip and unzip
### The large file "dataset.zip" is stored in Google Drive
```
* download [dataset.zip](https://drive.google.com/u/0/uc?id=1RYj9-eEC8S1SYyTRObo9J_XJhD2BAowB&export=download) and unzip 


## IR detection process
[src/main.py](https://github.com/genemine/DeepRetention/blob/master/src/main.py) is the entire detection pipeline. 
The specific components of pipeline are briefly described below.
* Step 1: get all intron from genome annotation file (gtf).
* Step 2: get read sequence from genome sequence file (fasta) and use doc2vec to convert into embedding.
* Step 3: get read coverage sequence from RNA-Seq (bam).
>Step 2 and step 3 can be executed in parallel
* Step 4: IR detection based on read sequence embedding, read coverage sequence and length of intron.

```
'''
Way1: Generate all the required files for DeepRetention from the original xx.bam.
'''
python src/main.py -b xxx.bam -g xxx.gtf -a xxx.fa -m model/DeepRetention_xxx -d doc2vec/doc2vec_xxx.pkl


'''
Way2: Directly detect IR by specifying the generated files.
(PS: Take the test data in the dataset dir as an example)
'''
python src/model.py --bed dataset/GM12878S.dataset --cov dataset/GM12878S.CoverageSeq.pkl --read dataset/GM12878S.ReadSeq_human.pkl --model model/DeepRetention_Human --out dataset/GM12878S.res
```
