#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import time
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='pipeline of DeepRetention')
parser.add_argument('-b','--bam', required=True, help='RNA-Seq sequecening data')
parser.add_argument('-g','--gtf', required=True, help='genome annotation file')
parser.add_argument('-a','--fasta', required=True, help='genome fasta file')
parser.add_argument('-m','--model', default='../model/DeepRetention_Human', help='DeepRetention dir')
parser.add_argument('-d','--doc2vec', default='../doc2vec/doc2vec_human.pkl', help='doc2vec dir')
parser.add_argument('-c','--cov', default="CoverageSeq.pkl", help='CoverageSeq sequence pkl')
parser.add_argument('-r','--read', default="ReadSeq.pkl", help='read sequence pkl')
parser.add_argument('-k','--kmer', default=3, help='word length')
parser.add_argument('-o','--out', default='./DeepRetention_RES', help='all output file in this folder')
args = parser.parse_args()
bam_file = args.bam
gtf_file = args.gtf
fasta_file = args.fasta
model = args.model
doc2vec = args.doc2vec
convSeq = args.cov
readSeq = args.read
kmer = int(args.kmer)
out_dir = args.out

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

src_dir = os.path.dirname(sys.argv[0])
bam_id = bam_file.split("/")[-1].split(".bam")[0]

# bam index
if not os.path.exists(bam_file+".bai"):
    os.system("samtools index {}".format(bam_file))

#######################################
# 1. get 'intron bed' from gtf
# input: gtf_file
# output: xxx.intron.bed, without header, columns=["chr","start","end","ENST","ENSG","strand"]
# tools: gtftools
#######################################
print("step1: get intron bed from gtf")

intron_bed=os.path.join(out_dir, "{}.intron.bed".format(bam_id))
if not os.path.exists(intron_bed):
    print("-- 1.producing {}.intron.bed........".format(bam_id))
    cmd = "python {}/gtftools.py -i {}.tmp {}".format(src_dir, intron_bed, gtf_file)
    os.system(cmd)
    # drop duplicates
    cmd = "cut -f1,2,3,5 {}.tmp | sort | uniq > {}".format(intron_bed, intron_bed)
    os.system(cmd)
    os.remove(intron_bed+".tmp")
else:
    print("{}.intron.bed already exist.".format(bam_id))
    
print("step1 done.")


#######################################
# 2. get 'read sequence'
# input: intron_bed, fasta_file
# output: ReadSeq.pkl
# tools: bedtools, getReadSeq.py
#######################################
print("step2: get read sequence")

readSeq = os.path.join(out_dir, readSeq)
if not os.path.exists(readSeq):
    print("-- 2.producing read sequence......")
    cmd = "python {}/getReadSeq.py --bed {} --fasta {} --doc2vec {} --out {} &".format(src_dir, intron_bed, fasta_file, doc2vec, readSeq)
    os.system(cmd)
else:
    print("ReadSeq.pkl already exist.")


#######################################
# 3. get 'read coverage sequence'
# input: intron_bed, bam_file
# output: chrom_dir, CoverageSeq.pkl
# tools: samtools, getReadCoverageSeq.py
#######################################
print("step3: get read coverage sequence")

convSeq = os.path.join(out_dir, convSeq)
if not os.path.exists(convSeq):
    print("-- 3.extracting read coverage sequence from bam......")
    cmd = "python {}/getReadCoverageSeq.py --bed {} --bam {} --out {}".format(src_dir, intron_bed, bam_file, convSeq)
    os.system(cmd)
else:
    print("CoverageSeq.pkl already exist.")


while not os.path.exists(convSeq) or not os.path.exists(readSeq):
    print("wait for {} produce, sleep 6 min zZ...".format(readSeq if os.path.exists(convSeq) else convSeq))
    time.sleep(360)

print("step2 done.")
print("step3 done.")
    
    
#######################################
# 4. use model to predict
# input: model, doc2vec, intron_bed
# output: xx_intron_retention.res
# tools: model.py
#######################################
print("step4: model predict")
    
print("-- 4.model predicting......")
intron_res = os.path.join(out_dir, "{}_intron_retention.res".format(bam_id))
cmd = "python {}/model.py --bed {} --cov {} --read {} --model {} --out {}".format(src_dir, intron_bed, convSeq, readSeq, model, intron_res)
os.system(cmd)

print("step4 done")