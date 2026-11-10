import numpy as np
import torch
from load_data import load_raw
def prepare_data(data_dir='data'):
    raw = load_raw(data_dir)
    train_df, test_df = raw['train'], raw['test']
    X_all = train_df.iloc[:, :187].values.astype(np.float32)
    y_all = train_df.iloc[:, 187].values.astype(np.int64)
    X_te = test_df.iloc[:, :187].values.astype(np.float32)
    y_te = test_df.iloc[:, 187].values.astype(np.int64)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_df))
    n_val = int(0.1 * len(train_df))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (
        torch.tensor(X_all[train_idx], dtype=torch.float32),
        torch.tensor(y_all[train_idx], dtype=torch.long),
        torch.tensor(X_all[val_idx], dtype=torch.float32),
        torch.tensor(y_all[val_idx], dtype=torch.long),
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_te, dtype=torch.long)
    )
if __name__ == '__main__':
    splits = prepare_data()
    print('Prepared tensors:', [s.shape for s in splits])
