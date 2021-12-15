#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import pandas as pd
from multiprocessing.pool import Pool

parser = argparse.ArgumentParser(description='get read coverage sequence')
parser.add_argument('-f','--bed', required=True, help='intron coordinate bed, read coverage sequence append to it')
parser.add_argument('-b','--bam', required=True, help='sequenc file xx.bam')
parser.add_argument('-o','--out', default="ConveraeSeq.pkl", help='save extracted converage sequence')
parser.add_argument('-w','--worker', default=10, help='Number of parallel processes')
parser.add_argument('-e','--eLen', default=40, help='the read length taken from exon')
parser.add_argument('-i','--iLen', default=40, help='the read length taken from intron')
parser.add_argument('-c','--cLen', default=40, help='the read length taken from intron center')
args = parser.parse_args()

intron_coord_file = args.bed
bam = args.bam
out = args.out
exonLen = int(args.eLen)
intronLen = int(args.iLen)
centerLen = int(args.cLen)
workers = int(args.worker)

def extractCoverageSeq(val_tuple):
    chrom, coord_s, coord_e = val_tuple[0:3]
    chrom, coord_s, coord_e = str(chrom), int(coord_s), int(coord_e)

    intervel = (coord_e-coord_s)//(7-1)
    ss, se = coord_s-exonLen, coord_s+intronLen
    cmd = "samtools depth -aa -r {}:{}-{} {} | cut -f3".format(chrom, ss, se, bam)
    cmd_res = os.popen(cmd)
    tmp_s = list(map(lambda x: int(x.strip()), cmd_res))
    for p in range(1,7-1):
        cs, ce = coord_s+(intervel*p)-centerLen, coord_s+(intervel*p)+centerLen
        cmd = "samtools depth -aa -r {}:{}-{} {} | cut -f3".format(chrom, cs, ce, bam)
        cmd_res = os.popen(cmd)
        tmp_c = list(map(lambda x: int(x.strip()), cmd_res))
        tmp_s = tmp_s + tmp_c

    es, ee = coord_e-intronLen, coord_e+exonLen
    cmd = "samtools depth -aa -r {}:{}-{} {} | cut -f3".format(chrom, es, ee, bam)
    cmd_res = os.popen(cmd)
    tmp_e = list(map(lambda x: int(x.strip()), cmd_res))

    tmp_bpCount = tmp_s + tmp_e
    return tmp_bpCount


if __name__ == "__main__":
    
    intron_coord = pd.read_csv(intron_coord_file,header=None,sep="\t",low_memory=False)
    with Pool(processes=workers) as pool:
        read_count_seq = pool.map(extractCoverageSeq, intron_coord.values)
    pd.DataFrame(read_count_seq).to_pickle(out)