from tqdm import tqdm
import tensorflow as tf

def load_size(path):
    with open(path) as fd:
        return int(fd.read())

def save_size(size, path):
    with open(path, 'w') as fd: 
        fd.write(str(size))

def get_reg_input(x):
    return ({'ids1': x['ids'][:,:256], 
             'ids2': x['ids'][:,256:]}, x['pct_ranks'])

def get_match_input(x):
    return {'anc_ids': x['mark_ids'][x['anc_idx']], 
            'pos_ids': x['code_ids'][x['pos_idx']], 
            'neg_ids': x['code_ids'][x['neg_idx']], 
            'margins': x['margins']}

def encode_triplet(anc_ids, pos_ids, neg_ids, margin):
    anc_ids = tf.train.Int64List(value=anc_ids)
    pos_ids = tf.train.Int64List(value=pos_ids)
    neg_ids = tf.train.Int64List(value=neg_ids)
    margins = tf.train.FloatList(value=[margin])
    example = tf.train.Example(
        features=tf.train.Features(feature={
            'anc_ids': tf.train.Feature(int64_list=anc_ids),
            'pos_ids': tf.train.Feature(int64_list=pos_ids),
            'neg_ids': tf.train.Feature(int64_list=neg_ids),
            'margins': tf.train.Feature(float_list=margins)}))
    return example.SerializeToString()

def decode_triplet(record):
    features = {
        'anc_ids': tf.io.FixedLenFeature([128], dtype=tf.int64),
        'pos_ids': tf.io.FixedLenFeature([128], dtype=tf.int64),
        'neg_ids': tf.io.FixedLenFeature([128], dtype=tf.int64),
        'margins': tf.io.FixedLenFeature([1], dtype=tf.float32)}
    return tf.io.parse_single_example(record, features)

def save_triplets(data, tfrec_path):
    with tf.io.TFRecordWriter(tfrec_path) as writer:
        triplets_indices = range(len(data['anc_ids']))
        for i in tqdm(triplets_indices, mininterval=1):
            encoded = encode_triplet(
                data['anc_ids'][i], data['pos_ids'][i], 
                data['neg_ids'][i], data['margins'][i])
            writer.write(encoded)

def get_dataset(data=None, paths=None, decode_fn=None, 
                shuffled=False, buf_size=2048, seed=0, 
                repeated=False, batch_size=64, strategy=None):

    if paths:
        auto = tf.data.experimental.AUTOTUNE
        dataset = tf.data.TFRecordDataset(paths, num_parallel_reads=auto)
        dataset = dataset.map(decode_fn, num_parallel_calls=auto)
    else: dataset = tf.data.Dataset.from_tensor_slices(data)

    if shuffled: dataset = dataset.shuffle(buf_size, seed=seed)
    if repeated: dataset = dataset.repeat()
    if batch_size: dataset = dataset.batch(batch_size)
    if paths: dataset = dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
    if strategy: dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset