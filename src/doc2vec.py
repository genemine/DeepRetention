#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import argparse
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

parser = argparse.ArgumentParser(description='construct doc2vec model')
parser.add_argument('-g','--gtf', required=True, help='genome annotation file')
parser.add_argument('-a','--fasta', required=True, help='genome fasta file')
parser.add_argument('-k','--kmer', default=3, help='word length')
parser.add_argument('-o','--output', default="doc2vec.pkl", help='output doc2vec name')
args = parser.parse_args()
genome_gtf = args.gtf
genome_fa = args.fasta
doc2vec = args.output
kmer = int(args.kmer)

def extract_seq(c):
    return not c.startswith(">")

def getKmer(seq):
    seq = seq.strip()
    length = len(seq)
    res = []
    for i in range(length-kmer+1):
        res.append(seq[i:i+kmer])
    return res

if __name__ == "__main__":

    '''1. extract the chromosome coordinates of all genes from the genome annotation file'''
    output_file = "./allGene.position.{}.tmp".format(doc2vec)
    if os.path.exits(output_file):
        extract_gene_from_gtf_cmd = "bash ./extractGeneFromGTF.sh {} {}".format(genome_gtf, output_file)
        os.system(extract_gene_from_gtf_cmd)

    '''2. according to extracted BED, extract sequences from genome fasta'''
    get_bp_cmd = "bedtools getfasta -fi {} -bed {}".format(genome_fa, output_file)
    cmd_res = os.popen(get_bp_cmd)
    bpSeq = list(map(lambda x: x.strip(), filter(extract_seq, cmd_res)))
    bpSeq = pd.DataFrame(bpSeq)


    '''3.  decompose read sequences for training doc2vec model'''
    bpSeq = bpSeq.apply(getKmer)
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(bpSeq[0])]
    model = Doc2Vec(documents, vector_size=100, window=5, min_count=3, epochs=15, negative=5, workers=10)
    model.save(doc2vec)