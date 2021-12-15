#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import argparse
import pandas as pd
from multiprocessing.pool import Pool
from gensim.models.doc2vec import Doc2Vec

parser = argparse.ArgumentParser(description='get read sequence')
parser.add_argument('-b','--bed', required=True, help='intron coordinate bed, read sequence append to it')
parser.add_argument('-f','--fasta', required=True, help='genome fasta sequence')
parser.add_argument('-o','--out', default="ReadSeq.pkl", help='save extracted read sequence')
parser.add_argument('-d','--doc2vec', default='../doc2vec/doc2vec_human.pkl', help='doc2vec dir')
parser.add_argument('-k','--kmer', default=3, help='word length (default 3), keep with doc2vec')
parser.add_argument('-w','--worker', default=10, help='multiprocess, default 10 process')
args = parser.parse_args()

genome_fa = args.fasta
intron_coord_file = args.bed
out = args.out
doc2vec = args.doc2vec
kmer = int(args.kmer)
worker = int(args.worker)

def filterSeq(seq):
    return not seq.startswith(">")

def getKmer(seq):
    seq = seq.strip()
    length = len(seq)
    res = []
    for i in range(length-kmer+1):
        res.append(seq[i:i+kmer])
    return doc2vec.infer_vector(res)


if __name__ == "__main__":
    intron_coord = pd.read_csv(intron_coord_file,header=None,sep="\t",low_memory=False)
    doc2vec = Doc2Vec.load(doc2vec)
    
    get_bp_cmd = "bedtools getfasta -fi {} -bed {}".format(genome_fa, intron_coord_file)
    cmd_res = os.popen(get_bp_cmd)
    
    if not cmd_res.readline():
        print("cmd: bedtools getfasta fail! please check your bed -- {} !".format(intron_coord_file))
        sys.exit()
        
    bp_seq = list(map(lambda x: x.strip(), filter(filterSeq, cmd_res)))
    with Pool(processes=worker) as pool:
        bpEmb = pool.map(getKmer, bp_seq)
    pd.DataFrame(bpEmb).to_pickle(out)