#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Extract per-base epigenomic signal around TSS±3kb and TTS±2kb for each gene,
# Usage: python get_epi.py -g AK58.gff3 -p YP.norm.bedGraph YP-K27ac.norm.bedGraph YP-K27.norm.bedGraph YP-K36.norm.bedGraph YP-K4.norm.bedGraph -o epi

import os
import argparse
from collections import defaultdict
from intervaltree import IntervalTree

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract per-base epigenomic signal around TSS±3kb and TTS±2kb for each gene.")
    parser.add_argument("-g", "--gff", required=True,
                        help="GFF3 annotation file containing gene features (ID=gene:...).")
    parser.add_argument("-p", "--peaks", required=True, nargs="+", action="append",
                        help="List of one or more peak file paths per -p; can use multiple -p flags.")
    parser.add_argument("-o", "--out", required=True,
                        help="Output TSV file prefix.")
    args = parser.parse_args()
    # flatten list of lists into a single list
    args.peaks = [f for sublist in args.peaks for f in sublist]
    return args


def load_peaks(peak_files):
    """
    Build an IntervalTree for each peak file keyed by chromosome.
    Stores signal values in half-open intervals [start, end).
    """
    trees = []
    for fname in peak_files:
        tree = defaultdict(IntervalTree)
        with open(fname) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                chrom, start, end, value = parts[0], int(parts[1]), int(parts[2]), float(parts[3])
                tree[chrom].addi(start, end, value)
        trees.append(tree)
    return trees


def parse_gff3(gff_file):
    """
    Parse GFF3 and return list of tuples: (chrom, strand, tss, tts, gene_id).
    """
    genes = []
    opener = open
    if gff_file.endswith('.gz'):
        import gzip
        opener = lambda f: gzip.open(f, 'rt')
    with opener(gff_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            cols = line.strip().split('\t')
            if len(cols) < 9 or cols[2] != 'gene':
                continue
            chrom, _, _, start, end, _, strand, _, attr = cols
            start, end = int(start), int(end)
            gene_id = None
            for field in attr.split(';'):
                if field.startswith('ID='):
                    val = field.split('=', 1)[1]
                    gene_id = val.split('gene:')[-1]
                    break
            if gene_id:
                tss = start if strand == '+' else end
                tts = end if strand == '+' else start
                genes.append((chrom, strand, tss, tts, gene_id))
    return genes


def extract_signals(genes, trees):
    """
    For each gene and each peak tree, sample signal at each base in:
      - TSS window: [tss - 3000, tss + 3000) → 6000 bases
      - TTS window: [tts - 2000, tts + 2000) → 4000 bases
    Reverse combined vector for '-' strand to keep 5'→3'.
    Returns list of (gene_id, [vec1, vec2, ...])
    """
    results = []
    for chrom, strand, tss, tts, gene_id in genes:
        gene_vecs = []
        for tree in trees:
            ivt = tree.get(chrom, IntervalTree())
            sig = []
            # TSS ±3kb: exactly 6000 positions
            for pos in range(tss - 3000, tss + 3000):
                hits = ivt[pos]
                sig.append(max(iv.data for iv in hits) if hits else 0.0)
            # TTS ±2kb: exactly 4000 positions
            for pos in range(tts - 2000, tts + 2000):
                hits = ivt[pos]
                sig.append(max(iv.data for iv in hits) if hits else 0.0)
            if strand == '-':
                sig.reverse()
            gene_vecs.append(sig)
        results.append((gene_id, gene_vecs))
    return results


def write_output(results, peak_files, out_prefix):
    """
    Write TSV: gene_id \t peak1_vec \t peak2_vec ...
    Each vec is comma-separated 10000 values.
    """
    names = [os.path.splitext(os.path.basename(f))[0] for f in peak_files]
    out_file = f"{out_prefix}_signal_TSS±3k_TTS±2k.tsv"
    with open(out_file, 'w') as out:
        out.write('gene_id\t' + '\t'.join(names) + '\n')
        for gid, vecs in results:
            str_vecs = [','.join(map(str, v)) for v in vecs]
            out.write(gid + '\t' + '\t'.join(str_vecs) + '\n')
    print(f"Written output: {out_file}")


def main():
    args = parse_args()
    print("Loading peaks...")
    trees = load_peaks(args.peaks)
    print("Parsing GFF3...")
    genes = parse_gff3(args.gff)
    print(f"Loaded {len(genes)} genes")
    print("Extracting signals (10000 points per gene)...")
    results = extract_signals(genes, trees)
    print("Writing results...")
    write_output(results, args.peaks, args.out)
    print("Done.")

if __name__ == '__main__':
    main()
