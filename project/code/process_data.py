import numpy as np
from scipy.sparse import csr_matrix


def parse_config(config_path: str):
    config_vals = {}
    with open(config_path, "r") as f:
        for line in f:
            key, val = line.split()
            config_vals[key] = int(val)

    return (
        config_vals["N_TRAIN"],
        config_vals["N_DEV"],
        config_vals["D"],
        config_vals["C"],
    )


def load_sparseX(path: str, n_rows: int, n_cols: int):
    rows = []
    cols = []
    vals = []

    with open(path, "r") as f:
        for line in f:
            i_str, j_str, v_str = line.split()
            rows.append(int(i_str))
            cols.append(int(j_str))
            vals.append(float(v_str))

    return csr_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))


def load_RT(path: str):
    return np.loadtxt(path)