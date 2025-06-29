#!/usr/bin/env python3
"""
Six‑organ gene‑expression prediction & Integrated‑Gradients attribution
======================================================================

This script reproduces the **exact preprocessing windows** used during
(`spike.h5`, `leaf.h5`, `vl.h5`, `e5.seed.h5`, `e7.seed.h5`,
`e9.seed.h5`) receive inputs in the same shape they were trained on.

For each organ‑specific model the script will
  1. load the corresponding epigenomic table `<prefix>.epi.tsv`,
  2. crop sequence & epi vectors using the indices below,
  3. predict **log1p(TPM)** (no conversion to TPM), save CSV,
  4. run Integrated Gradients (IG) for genes in `--attrib_list`,
  5. output sequence‑line + epi‑heatmap figures.

Inputs  (CLI)
-------------
--seq            : genome **sequence** TSV from A01_get_seq.py
--epi_dir        : directory with six `<prefix>.epi.tsv` files
--predict_list   : gene IDs to predict (one per line)
--attrib_list    : gene IDs to explain (one per line)
--model_dir      : directory with six trained `.h5` models
--tpm_dir        : directory with `<prefix>.tpm` (optional, for merge)
--out_pred       : output folder for prediction CSVs   (default: ./pred_results)
--out_ig         : output folder for IG figures        (default: ./IG_results)
--ig_steps       : IG linear‑path steps (default: 100)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

# -----------------------------------------------------------------------------
# CONSTANTS — same preprocessing windows as training
# -----------------------------------------------------------------------------
SEQ_IDX = np.concatenate((np.arange(1000, 4500), np.arange(7500, 8200)))  # len=4200
EPI_OFFSETS = [0, 10000, 20000, 30000, 40000]
EPI_IDX = np.hstack([off + SEQ_IDX for off in EPI_OFFSETS])  # len=21000 (5*4200)
PREFIXES = ['spike', 'leaf', 'vl', '5-DAP-seed', '7-DAP-seed', '9-DAP-seed']

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def read_genes(path: str) -> List[str]:
    return [ln.strip() for ln in open(path) if ln.strip() and not ln.startswith('#')]


def one_hot(seq: str) -> np.ndarray:
    m = {'A':0,'C':1,'G':2,'T':3}
    arr = np.zeros((len(seq),4),dtype=np.float32)
    for i,b in enumerate(seq.upper()):
        if b in m: arr[i,m[b]] = 1.0
    return arr


def load_seq(seq_tsv: str, genes: set[str]) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(seq_tsv, sep='\t', dtype=str)
    df = df[df.gene_id.isin(genes)].reset_index(drop=True)
    order = df.gene_id.tolist()
    full = np.stack([one_hot(s) for s in df.sequence])  # (N,10000,4)
    crop = full[:, SEQ_IDX, :]  # (N,4200,4)
    return order, crop


def load_epi(epi_tsv: str, order: List[str]) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(epi_tsv, sep='\t', dtype=str).set_index('gene_id').loc[order].reset_index()
    tracks = [c for c in df.columns if c!='gene_id']
    vals = []
    for row in df[tracks].itertuples(index=False):
        vec = []
        for cell in row: vec.extend(map(float, cell.split(',')))
        vals.append(vec)
    arr = np.array(vals, dtype=np.float32)
    scaled = MinMaxScaler().fit_transform(arr).astype(np.float32)
    crop = scaled[:, EPI_IDX]
    return crop, tracks

# -----------------------------------------------------------------------------
# Prediction & IG
# -----------------------------------------------------------------------------

def predict_log(model: tf.keras.Model, seq: np.ndarray, epi: np.ndarray) -> np.ndarray:
    seq_in = seq[..., np.newaxis]
    logp = model.predict([epi, seq_in], batch_size=256, verbose=0).flatten()
    return logp


def IG(model: tf.keras.Model,
       seq_s: np.ndarray,  # (1,4200,4,1)
       epi_s: np.ndarray,  # (1,21000)
       steps: int) -> Tuple[np.ndarray, np.ndarray]:
    t_seq = tf.convert_to_tensor(seq_s, tf.float32)
    t_epi = tf.convert_to_tensor(epi_s, tf.float32)
    b_seq = tf.zeros_like(t_seq)
    b_epi = tf.zeros_like(t_epi)
    d_seq = t_seq - b_seq
    d_epi = t_epi - b_epi
    g_seq = tf.zeros_like(t_seq)
    g_epi = tf.zeros_like(t_epi)

    for a in np.linspace(0,1,steps,dtype=np.float32):
        i_seq = b_seq + a*d_seq
        i_epi = b_epi + a*d_epi
        with tf.GradientTape() as tape:
            tape.watch([i_epi, i_seq])
            out = model([i_epi, i_seq], training=False)
        grad_e, grad_s = tape.gradient(out, [i_epi, i_seq])
        g_seq += grad_s
        g_epi += grad_e

    attr_s = (g_seq/steps)*d_seq
    attr_e = (g_epi/steps)*d_epi
    return attr_s.numpy().squeeze(), attr_e.numpy().squeeze()

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot(gid: str, a_s: np.ndarray, a_e: np.ndarray, tracks: List[str], out: Path):
    L = a_s.shape[0]
    seq_imp = a_s.sum(axis=1)  # 保留正负值，显示实际归因方向
    epi_mat = a_e.reshape((len(tracks), L))
    h = 2 + 0.4*len(tracks)
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,h), gridspec_kw={'height_ratios':[1, len(tracks)*0.4]})
    ax1.plot(seq_imp, lw=0.6); ax1.set_title(f'{gid} Seq IG')
    ax1.set_xlim(0,L)
    if sns:
        sns.heatmap(epi_mat, ax=ax2, cmap='coolwarm', center=0)
    else:
        im=ax2.imshow(epi_mat, aspect='auto', cmap='coolwarm', vmin=-abs(epi_mat).max(), vmax=abs(epi_mat).max())
        plt.colorbar(im, ax=ax2)
    ax2.set_yticks(np.arange(len(tracks))+0.5); ax2.set_yticklabels(tracks)
    ax2.set_title('Epi IG')
    plt.tight_layout(); fig.savefig(out, dpi=300); plt.close()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq', required=True)
    p.add_argument('--epi_dir', required=True)
    p.add_argument('--predict_list', required=True)
    p.add_argument('--attrib_list', required=True)
    p.add_argument('--model_dir', required=True)
    p.add_argument('--tpm_dir', default=None)
    p.add_argument('--out_pred', default='pred_results')
    p.add_argument('--out_ig', default='IG_results')
    p.add_argument('--ig_steps', type=int, default=50)
    args = p.parse_args()

    preds = set(read_genes(args.predict_list))
    attrs = set(read_genes(args.attrib_list))
    allg = preds|attrs

    order, seq_crop = load_seq(args.seq, allg)
    Path(args.out_pred).mkdir(exist_ok=True)
    Path(args.out_ig).mkdir(exist_ok=True)

    for pre in PREFIXES:
        mf = Path(args.model_dir)/f'{pre}.h5'
        ef = Path(args.epi_dir)/f'{pre}.epi.tsv'
        if not mf.exists() or not ef.exists(): continue
        print('->',pre)
        mdl = tf.keras.models.load_model(mf,compile=False)
        epi_crop, tracks = load_epi(str(ef), order)

        logs = predict_log(mdl, seq_crop, epi_crop)
        df = pd.DataFrame({'gene_id':order,'Pred_log1p':logs})
        if args.tpm_dir:
            tfp=Path(args.tpm_dir)/f'{pre}.tpm'
            if tfp.exists():
                tdf=pd.read_csv(tfp,sep='\t',header=None,names=['gene_id','True_TPM'])
                df=df.merge(tdf,on='gene_id',how='left')
        df.to_csv(Path(args.out_pred)/f'{pre}_pred.csv',index=False)

        igf=Path(args.out_ig)/pre; igf.mkdir(exist_ok=True)
        idx = {g:i for i,g in enumerate(order)}
        for g in attrs:
            if g not in idx: continue
            i = idx[g]
            s = seq_crop[i][None,...,None]
            e = epi_crop[i][None,:]
            a_s, a_e = IG(mdl, s, e, args.ig_steps)
            #  CSV
            # seq IG： position, attribution
            seq_attr_df = pd.DataFrame({'position': np.arange(a_s.shape[0]), 'attribution': a_s.sum(axis=1)})
            seq_attr_df.to_csv(igf / f'{g}_seq_attr.csv', index=False)
            # epi IG：row=track, line=position
            epi_mat = a_e.reshape((len(tracks), a_e.shape[0] // len(tracks)))
            epi_attr_df = pd.DataFrame(epi_mat, index=tracks)
            epi_attr_df.to_csv(igf / f'{g}_epi_attr.csv', index=True)
            # all region pef
            out_pdf = igf / f'{g}.pdf'
            plot(g, a_s, a_e, tracks, out_pdf)
            # 200 bp bin
            L = a_s.shape[0]
            seq_imp_full = a_s.sum(axis=1)
            epi_mat_full = a_e.reshape((len(tracks), L))
            window = 200
            for start in range(0, L, window):
                end = min(start + window, L)
                fig, (ax1, ax2) = plt.subplots(
                    2, 1,
                    figsize=(12, 2 + 0.4 * len(tracks)),
                    gridspec_kw={'height_ratios': [1, len(tracks) * 0.4]}
                )
                # seq
                ax1.plot(seq_imp_full[start:end], lw=0.6)
                ax1.set_xlim(0, end - start)
                ax1.set_title(f'{g} Seq IG ({start}-{end})')
                # epi
                if sns:
                    sns.heatmap(
                        epi_mat_full[:, start:end], ax=ax2,
                        cmap='coolwarm', center=0,
                        cbar_kws={'label': 'Epi IG'}
                    )
                else:
                    im = ax2.imshow(
                        epi_mat_full[:, start:end], aspect='auto', cmap='coolwarm',
                        vmin=-np.max(np.abs(epi_mat_full[:, start:end])),
                        vmax=np.max(np.abs(epi_mat_full[:, start:end]))
                    )
                    plt.colorbar(im, ax=ax2, label='Epi IG')
                ax2.set_yticks(np.arange(len(tracks)) + 0.5)
                ax2.set_yticklabels(tracks)
                ax2.set_title(f'{g} Epi IG ({start}-{end})')
                plt.tight_layout()
                sub_pdf = igf / f'{g}_{start}_{end}.pdf'
                fig.savefig(sub_pdf, dpi=300)
                plt.close()

    # end per-organ loop

if __name__=='__main__':
    # ensure GPU memory growth
    for gd in tf.config.list_physical_devices('GPU'):
        try: tf.config.experimental.set_memory_growth(gd,True)
        except: pass
    # run main
    print("Starting prediction and IG attribution...")
    main()

