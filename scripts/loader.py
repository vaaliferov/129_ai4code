import os
import ujson
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

def load_notebook(path):
    id = os.path.basename(path)[:-5]
    with open(path) as fd: data = ujson.load(fd)
    df = pd.DataFrame.from_dict(data).reset_index()
    df = df.rename(columns={'index': 'cell_id'})
    return df.assign(id=id).reset_index(drop=True)

def load_notebooks(path, num):
    paths = glob(f'{path}/*json')
    paths = tqdm(sorted(paths[:num]), mininterval=1)
    return pd.concat([load_notebook(p) for p in paths])

def get_train_pct_ranks(df, orders_df):
    odf = (orders_df.set_index('id')['cell_order']
           .str.split().explode().reset_index()
           .rename(columns={'cell_order':'cell_id'}))
    grouped = odf.groupby('id', sort=False)
    odf['rank'] = grouped['cell_id'].cumcount() + 1
    odf['count'] = grouped['cell_id'].transform('count')
    c1 = ['id', 'cell_id']; c2 = c1 + ['rank', 'count']
    df = df[c1].merge(odf[c2], on=c1, how='left')
    return (df['rank'] / df['count']).values

def get_pct_ranks(df, group_cols):
    grouped = df.groupby(group_cols, sort=False)
    ranks = grouped['cell_id'].cumcount() + 1
    counts = grouped['cell_id'].transform('count')
    return (ranks / counts).values

def train_test_split(df, ancestors_df, test_size, seed):
    ancestors = df['id'].map(ancestors_df.set_index('id')['ancestor_id'].to_dict())
    group_shuffle = GroupShuffleSplit(1, test_size=test_size, random_state=seed)
    train_idx, valid_idx = next(group_shuffle.split(df, groups=ancestors))
    return df.iloc[train_idx].copy(), df.iloc[valid_idx].copy()