import pandas as pd
from bisect import bisect
from loader import get_pct_ranks

def get_cell_order(df):
    df = df.groupby('id', sort=False)
    return df['cell_id'].apply(list)

def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions

def kendall_tau(ground_truth, predictions):
    total_2max, total_inversions = 0, 0
    for gt, pred in zip(ground_truth, predictions):
        n, ranks = len(gt), [gt.index(x) for x in pred] 
        total_inversions += count_inversions(ranks)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max

def evaluate(df, reg_ranks, match_ranks, 
             rerank_match=True, reg_coef=1.0, match_coef=1.0):
    
    df = df.copy()
    df['reg_rank'] = df['pct_rank']
    df['match_rank'] = df['pct_rank']
    m = df['cell_type'] == 'markdown'
    df.loc[m,'reg_rank'] = reg_ranks
    df.loc[m,'match_rank'] = match_ranks
    
    if rerank_match:
        df = df.sort_values(['id','match_rank'])
        df['match_rank'] = get_pct_ranks(df, 'id')
    
    df['ensemble_rank'] = (df['reg_rank'] * reg_coef + \
                           df['match_rank'] * match_coef) / 2
    
    o = get_cell_order(df.sort_values(['id','pct_rank']))
    r = get_cell_order(df.sort_values(['id','reg_rank']))
    m = get_cell_order(df.sort_values(['id','match_rank']))
    e = get_cell_order(df.sort_values(['id','ensemble_rank']))
    return (kendall_tau(o,r), kendall_tau(o,m), kendall_tau(o,e))

def submit(df, reg_ranks, match_ranks, 
           rerank_match=True, reg_coef=1.0, match_coef=1.0):
    
    df = df.copy()
    df['reg_rank'] = df['pct_rank']
    df['match_rank'] = df['pct_rank']
    m = df['cell_type'] == 'markdown'
    df.loc[m, 'reg_rank'] = reg_ranks
    df.loc[m, 'match_rank'] = match_ranks

    if rerank_match:
        df = df.sort_values(['id','match_rank'])
        df['match_rank'] = get_pct_ranks(df, 'id')
    
    df['ensemble_rank'] = (df['reg_rank'] * reg_coef + \
                           df['match_rank'] * match_coef) / 2

    sub_df = df.sort_values(by=['id','ensemble_rank']) \
        .groupby('id')['cell_id'].apply(lambda x: ' '.join(x)) \
        .reset_index().rename(columns={'cell_id': 'cell_order'})
    sub_df.to_csv('submission.csv', index=False)
    return pd.read_csv('submission.csv')