import io
import re
import math
import token
import random
import tokenize
import numpy as np
import pandas as pd
from tqdm import tqdm
from bisect import bisect_right

u16 = lambda x: np.array(x, dtype=np.uint16)
pad = lambda x, v, n: x + [v] * (n - len(x))
flatten = lambda x: [b for a in x for b in a]
find_le = lambda a, x: bisect_right(a, x) - 1

def concat_data(data):
    return {k: np.concatenate([x[k] for x in data]) for k in data[0]}

def extract_tokens(s):
    try:
        types = [token.STRING, token.COMMENT, token.NAME]
        gen = tokenize.generate_tokens(io.StringIO(s).readline)
        return ' '.join([t.string for t in gen if t.type in types])
    except: return s

def match_clean_code(s):
    s = re.sub('data:.*;base64,[^)>"]+', ' ', s, flags=re.S)
    return ' '.join(s.split())

def match_clean_mark(s):
    s = re.sub('data:.*;base64,[^)>"]+', ' ', s, flags=re.S)
    return ' '.join(s.split())

def reg_clean_code(s):
    s = extract_tokens(s)
    f = lambda x: f' {x[1]}{len(x.group())} '
    s = re.sub('([\*\=\-\~\.]){3,}', f, s, flags=re.S)
    s = re.sub('([^\s\'\"\(\)]{1000,})', ' ', s, flags=re.S)
    s = re.sub('data:.*;base64,[^)>"]+', ' ', s, flags=re.S)
    s = re.sub('[#\n\r_\'"]', ' ', s, flags=re.S)
    return ' '.join(s.split())

def reg_clean_mark(s):
    f = lambda x: f' {x[1]}{len(x.group())} '
    s = re.sub('([\*\#\=\-\~\.]){6,}', f, s, flags=re.S)
    s = re.sub('([^\s\'\"\(\)]{1000,})', ' ', s, flags=re.S)
    s = re.sub('data:.*;base64,[^)>"]+', ' ', s, flags=re.S)
    s = re.sub('&[A-Za-z]+;', ' ', s, flags=re.S)
    s = re.sub('</[^>]+>', ' ', s, flags=re.S)
    s = re.sub('<[^>]+>', ' ', s, flags=re.S)
    s = re.sub('[\n\r]', ' ', s, flags=re.S)
    return ' '.join(s.split())

def encode_sources(tokenizer, sources, max_len):
    if len(sources) == 0: return []
    p = dict(truncation=True, max_length=max_len)
    return tokenizer.batch_encode_plus(sources, **p)['input_ids']

def sample_codes(codes_ids, n):
    idx = np.linspace(0, len(codes_ids) - 1, n)
    idx = sorted(list({round(i) for i in idx}))
    return [codes_ids[i] for i in idx]

def get_total_len(ids):
    return sum([len(x) for x in ids])

def collate_ids(marks_ids, codes_ids, pad_token_id, max_len):
    
    if len(marks_ids) == 0:
        return np.zeros((0, max_len), dtype=np.uint16)
    
    if len(codes_ids) == 0:
        return u16([pad(x, pad_token_id, max_len) for x in marks_ids])
    
    nums = range(0, len(codes_ids) + 1)
    lens = [get_total_len(sample_codes(codes_ids, n)) for n in nums]
    nums, lens = np.argsort(lens).tolist(), np.sort(lens).tolist()
    nums = [nums[find_le(lens, max_len - len(x))] for x in marks_ids]
    sampled_codes_ids = [sample_codes(codes_ids, n) for n in nums]
    ids = [[m] + ctx for m, ctx in zip(marks_ids, sampled_codes_ids)]
    return u16([pad(flatten(x), pad_token_id, max_len) for x in ids])

def extract_reg_data(df, tokenizer, mark_max_len, code_max_len, max_len):
    ids, pct_ranks = [], []
    pad_id = tokenizer.pad_token_id
    for _, df in tqdm(df.groupby('id')):
        c = df['cell_type'] == 'code'
        m = df['cell_type'] == 'markdown'
        codes = [reg_clean_code(x) for x in df.loc[c, 'source']]
        marks = [reg_clean_mark(x) for x in df.loc[m, 'source']]
        pct_ranks.append(df.loc[m, 'pct_rank'].astype(np.float32))
        codes_ids = encode_sources(tokenizer, codes, code_max_len)
        marks_ids = encode_sources(tokenizer, marks, mark_max_len)
        ids.append(collate_ids(marks_ids, codes_ids, pad_id, max_len))
    return {'ids': np.vstack(ids), 'pct_ranks': np.hstack(pct_ranks)}

def extract_match_data(df, tokenizer, max_len, neg_num):
    
    unixcoder_prefix = [0,6,2]
    pad_id = tokenizer.pad_token_id
    code_offset, mark_offset, data = 0, 0, []

    for i, (_, df) in enumerate(tqdm(df.groupby('id'))):

        c = df['cell_type'] == 'code'
        m = df['cell_type'] == 'markdown'
        code_pos = df.loc[c, 'pct_rank'].tolist()
        mark_pos = df.loc[m, 'pct_rank'].tolist()
        codes = [match_clean_code(x) for x in df.loc[c, 'source']]
        marks = [match_clean_mark(x) for x in df.loc[m, 'source']]
        
        mark_ids = encode_sources(tokenizer, marks, max_len - 2)
        code_ids = encode_sources(tokenizer, codes, max_len - 2)
        mark_ids = [unixcoder_prefix + x[1:] for x in mark_ids]
        code_ids = [unixcoder_prefix + x[1:] for x in code_ids]
        mark_ids = [pad(x, pad_id, max_len) for x in mark_ids]
        code_ids = [pad(x, pad_id, max_len) for x in code_ids]
        
        anc_idx, neg_idx, pos_idx, margins = [], [], [], []
        mark_nb, code_nb = [i] * len(mark_pos), [i] * len(code_pos)

        if len(code_ids) == 0:
            code_ids = np.zeros((0, max_len), dtype=np.uint16)

        if len(mark_ids) == 0:
            mark_ids = np.zeros((0, max_len), dtype=np.uint16)

        if len(code_ids) > 1:
            for j in range(len(mark_ids)):
                d = lambda m, c: c - m if c > m else (m - c) * 2
                p = np.argmin([d(mark_pos[j], c) for c in code_pos])
                ns = [x for x in range(len(code_pos)) if x != p]
                ns = random.sample(ns, min(len(ns), neg_num))
                neg_idx.extend([n + code_offset for n in ns])
                anc_idx.extend([j + mark_offset] * len(ns))
                pos_idx.extend([p + code_offset] * len(ns))
                d = lambda c: abs(mark_pos[j] - code_pos[c])
                margins.extend([abs(d(p) - d(n)) for n in ns])

        data.append({
            'code_nb': np.array(code_nb, dtype=np.uint32), 
            'mark_nb': np.array(mark_nb, dtype=np.uint32), 
            'code_ids': np.array(code_ids, dtype=np.uint16), 
            'mark_ids': np.array(mark_ids, dtype=np.uint16), 
            'code_pos': np.array(code_pos, dtype=np.float32),
            'mark_pos': np.array(mark_pos, dtype=np.float32), 
            'margins': np.array(margins, dtype=np.float32), 
            'anc_idx': np.array(anc_idx, dtype=np.uint32), 
            'pos_idx': np.array(pos_idx, dtype=np.uint32), 
            'neg_idx': np.array(neg_idx, dtype=np.uint32)
        })
        
        mark_offset += len(mark_ids)
        code_offset += len(code_ids)
        
    return concat_data(data)