import pandas as pd
import numpy as np
import torch as th
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
from sklearn.metrics import roc_auc_score

from simufml.common.dataloader import DataSet, DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-2, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--no-shuffle', dest="shuffle", default=True, action='store_false')
    parser.add_argument('--no-cipher', dest="skip_cipher", default=False, action='store_true')
    parser.add_argument('--seed', dest='seed', default=None, type=int)
    parser.add_argument('--train-bs', dest='train_bs', default=16, type=int)
    parser.add_argument('--valid-bs', dest='valid_bs', default=64, type=int)
    args = parser.parse_args()
    return args


def csv2dl(
    active_csv,
    passive_csv,
    train_bs,
    valid_bs,
    tensor_data=False,
    rand_seed=None,
    intsect_col="id",
):

    passive_df = pd.read_csv(passive_csv)
    active_df = pd.read_csv(active_csv)

    # intersection
    intsect = set(passive_df[intsect_col].values).intersection(set(active_df[intsect_col].values))
    passive_df = passive_df.loc[passive_df[intsect_col].isin(intsect)].sort_values(by=intsect_col)
    active_df = active_df.loc[active_df[intsect_col].isin(intsect)].sort_values(by=intsect_col)

    # split
    passive_X = passive_df.values[:,1:].astype(np.float32)
    active_X = active_df.values[:,2:].astype(np.float32)
    active_Y = active_df.values[:,1].astype(np.float32)

    if tensor_data:
        passive_X = th.tensor(passive_X)
        active_X = th.tensor(active_X)
        active_Y = th.tensor(active_Y)

    np.random.seed(rand_seed)
    passive_train_X, passive_valid_X, \
    active_train_X, active_valid_X, \
    active_train_Y, active_valid_Y \
        = train_test_split(passive_X, active_X, active_Y, test_size=0.3)
    
    # to DataSet
    passive_train_ds = DataSet(X=passive_train_X)
    passive_valid_ds = DataSet(X=passive_valid_X)

    active_train_ds = DataSet(X=active_train_X, Y=active_train_Y)
    active_valid_ds = DataSet(X=active_valid_X, Y=active_valid_Y)

    # to DataLoader
    active_train_dl = DataLoader(active_train_ds, train_bs)
    active_valid_dl = DataLoader(active_valid_ds, valid_bs)

    passive_train_dl = DataLoader(passive_train_ds)
    passive_valid_dl = DataLoader(passive_valid_ds)

    active_fs = active_X.shape[1]
    passive_fs = passive_X.shape[1]

    return active_train_dl, active_valid_dl, passive_train_dl, passive_valid_dl, active_fs, passive_fs


def auc_func(batch_y, batch_predict_proba):
    batch_y = batch_y==1
    return roc_auc_score(batch_y, batch_predict_proba)


def abs_data_path(p):
    p = Path(p).absolute()
    assert 'simufml' in str(p), f"The project base 'simufml' must be in the path."
    while p.name!='simufml':
        p = p.parent
    return p/'demo/data'