#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, gzip, logging, sys, os
import numpy as np, pandas as pd, h5py, tensorflow as tf

# 
SEQ_LEN = 10_000
SEQ_IDX = np.concatenate((np.arange(1000, 4500), np.arange(7500, 8200)))
_ATAC  = np.concatenate((np.arange(1000,  4500),  np.arange(7500,  8200)))
_K27ac = np.concatenate((np.arange(11000,14500),  np.arange(17500,18200)))
_K27   = np.concatenate((np.arange(21000,24500),  np.arange(27500,28200)))
_K36   = np.concatenate((np.arange(31000,34500),  np.arange(37500,38200)))
_K4    = np.concatenate((np.arange(41000,44500),  np.arange(47500,48200)))
EPI_IDX = np.concatenate((_ATAC, _K27ac, _K27, _K36, _K4))
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Predict variant effect for ONE tissue")
    p.add_argument("--vcf",        required=True)
    p.add_argument("--gff",        required=True)
    p.add_argument("--seq-tsv",    required=True)
    p.add_argument("--epi-h5",     required=True)
    p.add_argument("--gene-list",  required=True)
    p.add_argument("--model",      required=True, help="model .h5 files")
    p.add_argument("--tissue",     default=None,  help="tissue name")
    p.add_argument("--out-prefix", required=True)
    return p.parse_args()

def load_gff(gff):
    info = {}
    op = gzip.open if gff.endswith(".gz") else open
    with op(gff, "rt") as fh:
        for ln in fh:
            if ln.startswith("#"): continue
            c = ln.rstrip().split("\t")
            if len(c) < 9 or c[2] != "gene": continue
            chrom, start, end, strand, attrs = c[0], int(c[3]), int(c[4]), c[6], c[8]
            gid = dict(x.split("=",1) for x in attrs.split(";") if "=" in x).get("ID","")
            gid = gid.split("gene:")[-1]
            if not gid: continue
            tss, tts = (start, end) if strand == "+" else (end, start)
            info[gid] = (chrom, strand, tss, tts)
    return info

def load_seq_tsv(path, genes):
    d={}
    with open(path) as fh:
        fh.readline()
        for ln in fh:
            gid, seq = ln.rstrip().split("\t")
            if gid in genes: d[gid] = seq.upper()
    return d

def load_epi(h5, gene_txt, genes):
    order = pd.read_csv(gene_txt, header=None)[0].tolist()
    idx = {g:i for i,g in enumerate(order) if g in genes}
    d={}
    with h5py.File(h5) as f:
        ds = f["dataset_1"]
        for g,i in idx.items():
            d[g]=ds[i][EPI_IDX].astype("float32")
    return d

def parse_vcf(vcf):
    op = gzip.open if vcf.endswith(".gz") else open
    with op(vcf,"rt") as fh:
        for ln in fh:
            if ln.startswith("#") or not ln.strip(): continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts)<5: continue
            chrom,pos,_id,ref,alt_field=parts[:5]
            try: pos=int(pos)
            except ValueError: continue
            for alt in alt_field.split(","):
                alt=alt.strip()
                if alt and alt!=".":
                    yield chrom,pos,ref.upper(),alt.upper()

def map_var(info, chrom, pos):
    for gid,(g_chr,strand,tss,tts) in info.items():
        if g_chr!=chrom: continue
        r1s,r1e = tss-3000, tss+3000
        r2s,r2e = tts-2000, tts+2000
        if strand=="+":
            if r1s<=pos<r1e: idx=pos-r1s
            elif r2s<=pos<r2e: idx=6000+(pos-r2s)
            else: continue
        else:
            if r1s<=pos<r1e: idx=5999-(pos-r1s)
            elif r2s<=pos<r2e: idx=6000+3999-(pos-r2s)
            else: continue
        yield gid,idx,strand

def one_hot(seq):
    arr=np.zeros((len(seq),4),dtype="float32")
    for i,b in enumerate(seq):
        if b=="A": arr[i,0]=1
        elif b=="C": arr[i,1]=1
        elif b=="G": arr[i,2]=1
        elif b=="T": arr[i,3]=1
    return arr

def apply_variant(seq, idx0, ref, alt, strand):
    if strand=="-":
        ref = ref.translate(COMP)[::-1]
        alt = alt.translate(COMP)[::-1]
    mut = seq[:idx0]+alt+seq[idx0+len(ref):]
    if len(mut)>SEQ_LEN: mut=mut[:SEQ_LEN]
    elif len(mut)<SEQ_LEN: mut += "N"*(SEQ_LEN-len(mut))
    return mut

def main():
    a=parse_args()
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])

    tissue = a.tissue or os.path.splitext(os.path.basename(a.model))[0]

    model=tf.keras.models.load_model(a.model, compile=False)
    logging.info(f"Model loaded for tissue: {tissue}")

    ginfo = load_gff(a.gff)

    # find variety
    rec=[]
    for chrom,pos,ref,alt in parse_vcf(a.vcf):
        for gid,idx,strand in map_var(ginfo,chrom,pos):
            rec.append((chrom,pos,ref,alt,gid,idx,strand))
    if not rec:
        logging.info("No variants within model window.")
        return
    genes={r[4] for r in rec}
    seqs = load_seq_tsv(a.seq_tsv, genes)
    epis = load_epi(a.epi_h5, a.gene_list, genes)

    out=[]
    for chrom,pos,ref,alt,gid,idx,strand in rec:
        seq_ref = seqs[gid]
        seq_alt = apply_variant(seq_ref, idx, ref, alt, strand)
        ref_oh  = one_hot(seq_ref)[SEQ_IDX][:,:,None]
        alt_oh  = one_hot(seq_alt)[SEQ_IDX][:,:,None]
        epi     = epis[gid]

        ref_pred = float(model.predict([epi[None,:], ref_oh[None,:,:,:]], verbose=0)[0])
        alt_pred = float(model.predict([epi[None,:], alt_oh[None,:,:,:]], verbose=0)[0])

        out.append(dict(
            chrom=chrom,pos=pos,gene_id=gid,strand=strand,
            ref=ref,alt=alt,tissue=tissue,
            log1p_ref=ref_pred,log1p_alt=alt_pred,
            delta_log1p=alt_pred-ref_pred,
            TPM_ref=np.expm1(ref_pred),
            TPM_alt=np.expm1(alt_pred),
            delta_TPM=np.expm1(alt_pred)-np.expm1(ref_pred)
        ))

    out_file=f"{a.out_prefix}_{tissue}.tsv"
    pd.DataFrame(out).to_csv(out_file,sep="\t",index=False,float_format="%.6g")
    logging.info(f"Saved {len(out):,} predictions → {out_file}")

if __name__=="__main__":
    main()
