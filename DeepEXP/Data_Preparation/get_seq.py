#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Extract TSS±3 kb and TTS±2 kb sequences for every gene in a GFF file,
# then output as tab-delimited gene_id and concatenated 10 kb sequence.
# Usage: python get_seq.py -g genome.fa -a genome.gff3 -o genome

import argparse
from Bio import SeqIO
from Bio.Seq import Seq


def load_genome(fasta_path):
    """Load genome FASTA into a dict: {chrom: Seq}"""
    genome = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        genome[rec.id] = rec.seq.upper()
    return genome


def parse_gff3(gff_path):
    """Parse GFF3 and return list of (chrom, gene_id, start, end, strand)."""
    genes = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            cols = line.strip().split('\t')
            if len(cols) < 9 or cols[2] != 'gene':
                continue
            chrom, _, _, start, end, _, strand, _, attr = cols
            attrs = dict(item.split('=') for item in attr.split(';') if '=' in item)
            raw_id = attrs.get('ID', '')
            if raw_id.startswith('gene:'):
                gene_id = raw_id.split(':', 1)[1]
                genes.append((chrom, gene_id, int(start), int(end), strand))
    return genes


def get_flank(seq, center, ups, downs, pad_char='N'):
    """
    Extract ups upstream and downs downstream bases around 1-based center.
    Pads with `pad_char` at boundaries.
    Returns a Seq of length ups+downs.
    """
    L = len(seq)
    start0 = center - ups  # 0-based start (inclusive)
    end0 = center + downs  # exclusive
    real_start = max(start0, 0)
    real_end = min(end0, L)
    sub = seq[real_start:real_end]
    left_pad = pad_char * (0 - start0) if start0 < 0 else ''
    right_pad = pad_char * (end0 - L) if end0 > L else ''
    return Seq(left_pad) + sub + Seq(right_pad)


def main():
    parser = argparse.ArgumentParser(description="Extract gene_id and 10kb TSS±3kb+TTS±2kb sequences")
    parser.add_argument('-g', '--genome', required=True, help="Genome FASTA file")
    parser.add_argument('-a', '--gff', required=True, help="GFF3 annotation file")
    parser.add_argument('-o', '--out', required=True, help="Output prefix")
    args = parser.parse_args()

    genome = load_genome(args.genome)
    genes = parse_gff3(args.gff)
    print(f"Found {len(genes)} genes. Processing...")

    out_tsv = f"{args.out}_TSS±3k_TTS±2k_seq.tsv"
    with open(out_tsv, 'w') as out:
        out.write("gene_id\tsequence\n")
        for chrom, gid, start, end, strand in genes:
            seq = genome.get(chrom)
            if seq is None:
                print(f"WARNING: chromosome {chrom} not in FASTA, skipping {gid}")
                continue
            # determine TSS and TTS
            tss = start if strand == '+' else end
            tts = end   if strand == '+' else start
            # extract and pad
            seq_tss = get_flank(seq, tss, ups=3000, downs=3000)
            seq_tts = get_flank(seq, tts, ups=2000, downs=2000)
            if strand == '-':
                seq_tss = seq_tss.reverse_complement()
                seq_tts = seq_tts.reverse_complement()
            concat = seq_tss + seq_tts  # length = 6000+4000 = 10000
            out.write(f"{gid}\t{concat}\n")

    print(f"Finished writing: {out_tsv}")

if __name__ == '__main__':
    main()
