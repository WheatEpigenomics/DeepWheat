#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------#
# Dual-branch model: log1p(TPM) labels                                  #
# ----------------------------------------------------------------------#
import os, gc, json, logging, argparse
import numpy as np, pandas as pd, h5py, scipy.stats, optuna
import matplotlib.pyplot as plt
import tensorflow as tf, tensorflow_addons as tfa

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from tensorflow import keras
from tensorflow.keras import layers, backend as K
try:
    from tensorflow.keras.optimizers.legacy import Adam, RMSprop, SGD
except ImportError:  # TF-2.9+
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from optuna.integration import TFKerasPruningCallback

# 1. Command-line arguments
def parse_args():
    ap = argparse.ArgumentParser(
        description="Train dual-branch model with log1p(TPM) labels "
                    "and evaluate R2 / PCC in both log1p & TPM space.")
    ap.add_argument("--seq",        default="./tissue.seq.tsv",   help="seq TSV")
    ap.add_argument("--epi",        default="./tissue.epi.tsv",   help="epi TSV")
    ap.add_argument("--tpm",        default="./tissue.logp(TPM)", help="TPM file")
    ap.add_argument("--test2",      default="./4700_gene.txt",   help="test2 gene list")
    ap.add_argument("--out_prefix", default="run",               help="out prefix")
    ap.add_argument("--batch_search", type=int, default=128,     help="Optuna batch_size")
    ap.add_argument("--batch_train",  type=int, default=64,      help="Train batch_size")
    return ap.parse_args()

ARGS = parse_args()

# ========= load
SEQ_TSV   = ARGS.seq
EPI_TSV   = ARGS.epi
TPM_TSV   = ARGS.tpm
TEST2_TXT = ARGS.test2

SEQ_H5    = f"{ARGS.out_prefix}_seq.h5"
EPI_H5    = f"{ARGS.out_prefix}_epi.h5"
GENE_LIST = f"{ARGS.out_prefix}_gene_list.txt"

RAND_TXT  = f"{ARGS.out_prefix}_rand_idx.txt"
BEST_JSON = f"{ARGS.out_prefix}_best_params.json"
WEIGHTS_H5= f"{ARGS.out_prefix}_model.best.h5"
LOG_FILE  = f"{ARGS.out_prefix}_train.log"
FIG_DIR   = f"{ARGS.out_prefix}_figs"

BATCH_SIZE_SEARCH = ARGS.batch_search
BATCH_SIZE_TRAIN  = ARGS.batch_train
# 2. GPU
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

gpus = tf.config.list_physical_devices("GPU")
for _gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass
try:
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
except Exception:
    pass

# 3. load data
def prepare_h5():
    """ TSV → HDF5 """
    log.info("=== [Step 1] gene_id and HDF5 get ===")
    seq_df = pd.read_csv(SEQ_TSV, sep="\t", dtype=str)
    epi_df = pd.read_csv(EPI_TSV, sep="\t", dtype=str)
    tpm_df = pd.read_csv(TPM_TSV, sep="\t", header=0, dtype=str)
    if tpm_df.shape[1] < 2:
        raise ValueError("TPM file needs ≥ 2 columns")
    tpm_df.columns = ["gene_id", "TPM"]

    merged = seq_df.merge(epi_df, on="gene_id").merge(tpm_df, on="gene_id")
    log.info(f"obatin  {len(merged)} gene number")

    # ---------- one-hot seq ----------
    seqs = merged["sequence"].tolist()
    L = len(seqs[0])
    dna = np.zeros((len(seqs), L, 4), dtype="float16")
    for i, s in enumerate(tqdm(seqs, desc="one-hot 序列")):
        for j, b in enumerate(s.upper()):
            if   b == "A": dna[i, j, 0] = 1
            elif b == "C": dna[i, j, 1] = 1
            elif b == "G": dna[i, j, 2] = 1
            elif b == "T": dna[i, j, 3] = 1
    with h5py.File(SEQ_H5, "w") as f:
        f.create_dataset("dataset_2", data=dna, compression="gzip", chunks=True)

    # ---------- epi ----------
    epi_cols = [c for c in merged.columns if c not in ("gene_id", "sequence", "TPM")]
    epi_list = []
    for row in tqdm(merged[epi_cols].itertuples(index=False), desc="解析 epi"):
        vec = []
        for cell in row:
            vec.extend(map(float, cell.split(",")))
        epi_list.append(vec)
    epi = np.array(epi_list, dtype="float32")
    epi = MinMaxScaler().fit_transform(epi).astype("float32")
    with h5py.File(EPI_H5, "w") as f:
        f.create_dataset("dataset_1", data=epi, compression="gzip", chunks=True)

    pd.Series(merged["gene_id"]).to_csv(GENE_LIST, index=False, header=False)
    log.info("HDF5 and gene_list obatin")

def load_data():
    if not (os.path.exists(SEQ_H5) and os.path.exists(EPI_H5) and os.path.exists(GENE_LIST)):
        prepare_h5()

    genes = pd.read_csv(GENE_LIST, header=None)[0].tolist()
    with h5py.File(SEQ_H5, "r") as f:
        seq = f["dataset_2"][:]
    with h5py.File(EPI_H5, "r") as f:
        epi = f["dataset_1"][:]
    log.info(f"Load seq {seq.shape}, epi {epi.shape}")
    ATAC = np.concatenate((np.arange(1000, 4500), np.arange(7500, 8200)))
    K27a = np.concatenate((np.arange(11000, 14500), np.arange(17500, 18200)))
    K27  = np.concatenate((np.arange(21000, 24500), np.arange(27500, 28200)))
    K36  = np.concatenate((np.arange(31000, 34500), np.arange(37500, 38200)))
    K4   = np.concatenate((np.arange(41000, 44500), np.arange(47500, 48200)))
    epi_idx = np.concatenate((ATAC, K27a, K27, K36, K4))
    seq_idx = np.concatenate((np.arange(1000, 4500), np.arange(7500, 8200)))
    epi = epi[:, epi_idx]
    seq = seq[:, seq_idx, :]
    log.info(f"seq {seq.shape}, epi {epi.shape}")
    # ---------- target ----------
    tpm = pd.read_csv(TPM_TSV, sep="\t", header=0, dtype=str)
    tpm.columns = ["gene_id", "TPM"]
    tpm["TPM"] = tpm["TPM"].astype("float32")
    tpm["log1p_TPM"] = np.log1p(tpm["TPM"])
    tpm = tpm.set_index("gene_id")["log1p_TPM"]
    label = np.array([tpm[g] for g in genes], dtype="float32")

    # ---------- test2 ----------
    t2_ids = pd.read_csv(TEST2_TXT, header=None)[0].tolist()
    if t2_ids and str(t2_ids[0]).lower().startswith("gene"):
        t2_ids = t2_ids[1:]
    mask2 = np.array([g in set(t2_ids) for g in genes])

    genes = np.array(genes)
    epi2,  seq2,  lab2,  gene2  = epi[mask2],  seq[mask2],  label[mask2],  genes[mask2]
    epi_r, seq_r, lab_r, gene_r = epi[~mask2], seq[~mask2], label[~mask2], genes[~mask2]

    # ---------- random ----------
    if os.path.exists(RAND_TXT):
        order = pd.read_csv(RAND_TXT, header=None)[0].tolist()
    else:
        order = np.arange(len(epi_r))
        np.random.shuffle(order)
        pd.Series(order).to_csv(RAND_TXT, index=False, header=False)
    epi_r, seq_r, lab_r, gene_r = epi_r[order], seq_r[order], lab_r[order], gene_r[order]

    # ---------- split ----------
    tr_e, va_e, te_e = epi_r[:70000], epi_r[70000:92000], epi_r[92000:]
    tr_s, va_s, te_s = seq_r[:70000], seq_r[70000:92000], seq_r[92000:]
    tr_l, va_l, te_l = lab_r[:70000], lab_r[70000:92000], lab_r[92000:]
    tr_g, va_g, te_g = gene_r[:70000], gene_r[70000:92000], gene_r[92000:]

    sub_idx = np.random.choice(tr_e.shape[0], int(0.2 * tr_e.shape[0]), replace=False)
    sub_e, sub_s, sub_l = tr_e[sub_idx], tr_s[sub_idx], tr_l[sub_idx]

    log.info(f"train {len(tr_e)}, valid {len(va_e)}, test {len(te_e)}, test2 {len(epi2)}")
    return {
        "train": (tr_e, tr_s, tr_l, tr_g),
        "valid": (va_e, va_s, va_l, va_g),
        "test":  (te_e, te_s, te_l, te_g),
        "test2": (epi2, seq2, lab2, gene2),
        "sub":   (sub_e, sub_s, sub_l),
    }

# 4. Custom evaluation metric & ResBlock
def RMS(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def res_block(x, filters, kernel_size, L2, dropout_rate, activation):
    shortcut = x
    y = layers.Conv2D(filters, kernel_size, padding="same",
                      kernel_regularizer=keras.regularizers.l2(L2))(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation(activation)(y)
    y = layers.Dropout(dropout_rate)(y)
    y = layers.Conv2D(filters, kernel_size, padding="same",
                      kernel_regularizer=keras.regularizers.l2(L2))(y)
    y = layers.BatchNormalization()(y)
    if K.int_shape(shortcut)[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding="same",
                                 kernel_regularizer=keras.regularizers.l2(L2))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    y = layers.Add()([y, shortcut])
    y = layers.Activation(activation)(y)
    y = layers.Dropout(dropout_rate)(y)
    return y
# 5. Model building
def build_model(k1, k2, ksz, L2, dr, nrb, rksz, optimizer, act,
                epi_dim, seq_shape):
    # epi branch
    ei = keras.Input((epi_dim,), name="epi_input")
    x = layers.Reshape((5, epi_dim // 5, 1))(ei)
    x = layers.Permute((2, 1, 3))(x)
    x = layers.ZeroPadding2D(((10, 9), (0, 0)))(x)
    x = layers.Conv2D(k1, (ksz, 5), use_bias=False, padding="valid",
                      kernel_regularizer=keras.regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act)(x)
    x = layers.Dropout(dr)(x)
    x = layers.ZeroPadding2D(((5, 4), (0, 0)))(x)
    x = layers.Conv2D(k2, (ksz, 1), padding="valid",
                      kernel_regularizer=keras.regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(act)(x)
    x = layers.Dropout(dr)(x)
    x = layers.AveragePooling2D((5, 1), strides=5)(x)
    epi_out = x

    # seq branch
    si = keras.Input((seq_shape[0], seq_shape[1], 1), name="seq_input")
    y = layers.ZeroPadding2D(((10, 9), (0, 0)))(si)
    y = layers.Conv2D(k1, (ksz, seq_shape[1]), use_bias=False, padding="valid",
                      kernel_regularizer=keras.regularizers.l2(L2))(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation(act)(y)
    y = layers.Dropout(dr)(y)
    y = layers.ZeroPadding2D(((5, 4), (0, 0)))(y)
    y = layers.Conv2D(k2, (ksz, 1), padding="valid",
                      kernel_regularizer=keras.regularizers.l2(L2))(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation(act)(y)
    y = layers.Dropout(dr)(y)
    y = layers.AveragePooling2D((5, 1), strides=5)(y)
    seq_out = y

    # fusion + residual
    z = layers.Concatenate(axis=2)([epi_out, seq_out])
    for _ in range(nrb):
        z = res_block(z, k1, (rksz, rksz), L2, dr, act)
    z = layers.Flatten()(z)
    for d in (256, 64, 16):
        z = layers.Dense(d, kernel_regularizer=keras.regularizers.l2(L2))(z)
        z = layers.BatchNormalization()(z)
        z = layers.Activation(act)(z)
        z = layers.Dropout(dr)(z)
    o = layers.Dense(1, activation="linear",
                     kernel_regularizer=keras.regularizers.l2(L2))(z)
    o = layers.Lambda(lambda t: tf.squeeze(t, -1))(o)
    model = keras.Model([ei, si], o)
    model.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=["mse", "mae", "mape", RMS,
                 tfa.metrics.RSquare(name="r2")],
    )
    return model

# ----------------------------------------------------------------------#
# 6. Optuna                                                             #
# ----------------------------------------------------------------------#
def optuna_obj(trial, sub):
    e, s, l = sub
    p = dict(
        lr   = trial.suggest_float("lr", 1e-6, 1e-3, log=True),
        k1   = trial.suggest_int("k1", 32, 128),
        k2   = trial.suggest_int("k2", 64, 256),
        ksz  = trial.suggest_int("ksz", 10, 60),
        L2   = trial.suggest_float("L2", 1e-5, 1e-2, log=True),
        dr   = trial.suggest_float("dr", 0.1, 0.8),
        nrb  = trial.suggest_int("nrb", 2, 10),
        rksz = trial.suggest_int("rksz", 2, 7),
        opt  = trial.suggest_categorical("opt", ["Adam", "RMSprop", "SGD"]),
        act  = trial.suggest_categorical("act", ["relu", "elu", "selu", "tanh"]),
    )
    OptCls = {"Adam": Adam, "RMSprop": RMSprop, "SGD": SGD}[p["opt"]]
    optimizer = OptCls(learning_rate=p["lr"])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, va_idx in kf.split(e):
        model = build_model(
            p["k1"], p["k2"], p["ksz"], p["L2"], p["dr"],
            p["nrb"], p["rksz"], optimizer, p["act"],
            e.shape[1], s.shape[1:])
        train_ds = tf.data.Dataset.from_tensor_slices(
            ((e[tr_idx], s[tr_idx]), l[tr_idx])
        ).shuffle(buffer_size=4 * BATCH_SIZE_SEARCH)\
         .batch(BATCH_SIZE_SEARCH)\
         .prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices(
            ((e[va_idx], s[va_idx]), l[va_idx])
        ).batch(BATCH_SIZE_SEARCH)\
         .prefetch(tf.data.AUTOTUNE)
        pruning_cb = TFKerasPruningCallback(trial, "val_r2")
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_r2", patience=5, mode="max", restore_best_weights=True)
        try:
            model.fit(train_ds, validation_data=val_ds, epochs=20, verbose=0,
                      callbacks=[pruning_cb, early_stop])
        except tf.errors.ResourceExhaustedError:
            K.clear_session(); gc.collect()
            return float("-inf")
        r2_val = model.evaluate(val_ds, verbose=0)[-1]
        scores.append(r2_val)
        K.clear_session(); gc.collect()
    return np.mean(scores)

# 7. Core procedure
def main():
    data = load_data()
    tr_e, tr_s, tr_l, tr_g = data["train"]
    va_e, va_s, va_l, va_g = data["valid"]
    te_e, te_s, te_l, te_g = data["test"]
    t2_e, t2_s, t2_l, t2_g = data["test2"]
    sub_e, sub_s, sub_l    = data["sub"]

    # ---------- Optuna ----------
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    log.info("Optuna Start hyperparameter search ...")
    study.optimize(lambda t: optuna_obj(t, (sub_e, sub_s, sub_l)),
                   n_trials=100, show_progress_bar=True)
    best = study.best_params
    json.dump(best, open(BEST_JSON, "w"))
    log.info(f"Best params: {best}")

    # ---------- train model ----------
    OptCls = {"Adam": Adam, "RMSprop": RMSprop, "SGD": SGD}[best["opt"]]
    optimizer = OptCls(learning_rate=best["lr"])
    model = build_model(
        best["k1"], best["k2"], best["ksz"], best["L2"],
        best["dr"], best["nrb"], best["rksz"], optimizer,
        best["act"], tr_e.shape[1], tr_s.shape[1:])
    
    cbs = [
        keras.callbacks.ModelCheckpoint(f"{ARGS.out_prefix}_full_model.h5",monitor="val_r2",save_best_only=True,mode="max",save_weights_only=False),
        keras.callbacks.ReduceLROnPlateau(monitor="val_r2",factor=0.5,patience=5,min_lr=1e-7),
        keras.callbacks.EarlyStopping(monitor="val_r2",patience=40,mode="max",restore_best_weights=True)]

    gc.collect()
    hist = model.fit(
        [tr_e, tr_s], tr_l,
        validation_data=([va_e, va_s], va_l),
        epochs=1000,
        batch_size=BATCH_SIZE_TRAIN,
        callbacks=cbs, verbose=1)
    #model.load_weights(WEIGHTS_H5)
    
    # ---------- Model evaluation: load the best full model ----------
    model = tf.keras.models.load_model(
        f"{ARGS.out_prefix}_full_model.h5", compile=False )
    def eval_save(e, s, l_log, g, tag):
        pred_log = model.predict([e, s], batch_size=8).flatten()

        r2_log  = r2_score(l_log, pred_log)
        pcc_log, _ = scipy.stats.pearsonr(l_log, pred_log)
        true_tpm = np.expm1(l_log)
        pred_tpm = np.expm1(pred_log)
        r2_tpm   = r2_score(true_tpm, pred_tpm)
        pcc_tpm, _ = scipy.stats.pearsonr(true_tpm, pred_tpm)

        df = pd.DataFrame({
            "gene_id": g,
            "True_log1p": l_log,
            "Pred_log1p": pred_log,
            "True_TPM": true_tpm,
            "Pred_TPM": pred_tpm,
            "R2_log1p":  r2_log,
            "PCC_log1p": pcc_log,
            "R2_TPM":   r2_tpm,
            "PCC_TPM":  pcc_tpm,
        })
        out_csv = f"{ARGS.out_prefix}_{tag}_results.csv"
        df.to_csv(out_csv, index=False)
        log.info(f"{tag}: R2_log1p={r2_log:.3f}, PCC_log1p={pcc_log:.3f}, "
                 f"R2_TPM={r2_tpm:.3f}, PCC_TPM={pcc_tpm:.3f}  -> {out_csv}")

    eval_save(te_e,  te_s,  te_l,  te_g,  "test")
    eval_save(t2_e, t2_s, t2_l, t2_g, "test2")

    # ---------- Training curve ----------
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.figure(figsize=(18, 10))
    metrics = [("loss", "Loss(MSE)"), ("mse", "MSE"), ("mae", "MAE"),
               ("mape", "MAPE"), ("RMS", "RMSE"), ("r2", "R2")]
    for i, (k, title) in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        plt.plot(hist.history[k], label="train")
        plt.plot(hist.history["val_" + k], label="val")
        plt.title(title); plt.legend()
    plt.tight_layout()
    fig_file = f"{FIG_DIR}/training_history.pdf"
    plt.savefig(fig_file); plt.close()
    log.info(f"{fig_file}")

if __name__ == "__main__":
    main()



