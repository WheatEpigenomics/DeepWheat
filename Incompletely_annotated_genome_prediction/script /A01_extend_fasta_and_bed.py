#!/usr/bin/env python3
"""
Extend a FASTA sequence to 131 072 bp with Ns and
shift a BED file so coordinates remain correct.

Example:
    python extend_fasta_and_bed.py \
        --inseq  you_predict.fa \
        --inbed  you_predict.bed \
        --outseq model_input.fa \
        --outbed model_input.bed
"""
import argparse
from pathlib import Path
from textwrap import fill
from typing import Tuple

from Bio import SeqIO          # pip install biopython

TARGET_LEN = 131_072           # desired window length (128 KiB)


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def pad_sequence(seq: str, target_len: int) -> Tuple[str, int]:
    """Return padded sequence + left-padding size (bp)."""
    cur_len = len(seq)
    if cur_len > target_len:
        raise ValueError(
            f"Input sequence length {cur_len:,} > target {target_len:,} bp."
        )
    pad_total = target_len - cur_len
    left_pad  = pad_total // 2
    right_pad = pad_total - left_pad
    padded    = "N" * left_pad + seq + "N" * right_pad
    return padded, left_pad


def wrap_fasta(seq_id: str, seq: str, width: int = 60) -> str:
    """Return wrapped FASTA string."""
    return f">{seq_id}\n" + "\n".join(fill(seq, width).splitlines())


def shift_bed_line(line: str, offset: int) -> str:
    """Shift BED start/end by offset; keep other columns unchanged."""
    if not line or line.startswith("#"):
        return line.rstrip()
    parts = line.rstrip().split("\t")
    parts[1] = str(int(parts[1]) + offset)
    parts[2] = str(int(parts[2]) + offset)
    return "\t".join(parts)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend FASTA to 131 072 bp and adjust BED coordinates."
    )
    parser.add_argument("--inseq",  required=True, help="input FASTA file")
    parser.add_argument("--inbed",  required=True, help="input BED file")
    parser.add_argument("--outseq", required=True, help="output FASTA file")
    parser.add_argument("--outbed", required=True, help="output BED file")
    args = parser.parse_args()

    inseq  = Path(args.inseq)
    inbed  = Path(args.inbed)
    outseq = Path(args.outseq)
    outbed = Path(args.outbed)

    # -- Read FASTA (assume 1 record) ---------------------------------------
    record = next(SeqIO.parse(inseq, "fasta"))
    padded_seq, left_pad = pad_sequence(str(record.seq).upper(), TARGET_LEN)

    # -- Write extended FASTA ----------------------------------------------
    outseq.write_text(wrap_fasta(record.id, padded_seq))
    print(f"✅ FASTA written  ➜ {outseq}  ({TARGET_LEN:,} bp)")

    # -- Shift BED ----------------------------------------------------------
    with inbed.open() as src, outbed.open("w") as dst:
        for line in src:
            dst.write(shift_bed_line(line, left_pad) + "\n")
    print(f"✅ BED written    ➜ {outbed}  (shift +{left_pad} bp)")


if __name__ == "__main__":
    main()
