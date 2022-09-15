import tensorflow as tf
from transformers import TFAutoModel
from transformers import AdamWeightDecay, WarmUp
from tensorflow.keras.experimental import CosineDecay

def rowwise_cosine_similarity(a, b):
    norm_a, norm_b = tf.norm(a, axis=1), tf.norm(b, axis=1)
    return tf.reduce_sum(a * b, axis=1) / tf.multiply(norm_a, norm_b)

def pairwise_cosine_similarity(a, b):
    norm_a, norm_b = tf.norm(a, axis=1), tf.norm(b, axis=1)
    return tf.matmul(a, tf.transpose(b)) / tf.tensordot(norm_a, norm_b, 0)

def triplet_loss(anc_embs, pos_embs, neg_embs, margins):
    dn = rowwise_cosine_similarity(anc_embs, neg_embs)
    dp = rowwise_cosine_similarity(anc_embs, pos_embs)
    return tf.maximum(dn - dp + margins, 0)

def get_optimizer(learning_rate, warmup_steps, total_steps, weight_decay_rate):
    
    decay_steps = total_steps - warmup_steps
    schedule_fn = CosineDecay(learning_rate, decay_steps)
    lr_schedule = WarmUp(learning_rate, schedule_fn, warmup_steps)

    return AdamWeightDecay(
        learning_rate=lr_schedule, weight_decay_rate=weight_decay_rate, 
        exclude_from_weight_decay=('bias','LayerNorm.bias','LayerNorm.weight')
    )

def get_reg_model(bert_name_or_path, pad_token_id):
    
    bert = TFAutoModel.from_pretrained(bert_name_or_path)
    ids1 = tf.keras.layers.Input((256,), dtype=tf.int32, name='ids1')
    ids2 = tf.keras.layers.Input((256,), dtype=tf.int32, name='ids2')
    ids = tf.concat((ids1, ids2), axis=1)
    
    embs = bert({'input_ids': ids, 'attention_mask': ids != pad_token_id})[0]
    mask = tf.cast(tf.tile(tf.expand_dims(ids != pad_token_id, -1), (1,1,768)), tf.float32)
    embs = tf.reduce_sum(embs * mask, 1) / tf.maximum(tf.reduce_sum(mask, 1), 1e-9)
    embs = tf.math.l2_normalize(embs, 1)
    
    glorot = tf.keras.initializers.GlorotNormal()
    p = {'kernel_initializer': glorot, 'dtype': 'float32'}
    x = tf.keras.layers.Dense(384, activation='tanh', **p)(embs)
    x = tf.keras.layers.Dropout(rate=0.2, seed=0)(x)
    pct_ranks = tf.keras.layers.Dense(1, **p)(x)
    
    return tf.keras.Model(inputs=[ids1, ids2], outputs=pct_ranks)

def get_match_model(bert_name_or_path, batch_size, pad_token_id, from_pt):
    
    ids = tf.keras.layers.Input((128,), dtype=tf.int32)
    bert = TFAutoModel.from_pretrained(bert_name_or_path, from_pt=from_pt)
    embs = bert({'input_ids': ids, 'attention_mask': ids != pad_token_id})[0]
    mask = tf.cast(tf.tile(tf.expand_dims(ids != pad_token_id, -1), (1,1,768)), tf.float32)
    embs = tf.reduce_sum(embs * mask, 1) / tf.maximum(tf.reduce_sum(mask, 1), 1e-9)
    embedder = tf.keras.Model(inputs=ids, outputs=tf.math.l2_normalize(embs, 1))
    
    anc_ids = tf.keras.layers.Input((128,), dtype=tf.int32, name='anc_ids')
    pos_ids = tf.keras.layers.Input((128,), dtype=tf.int32, name='pos_ids')
    neg_ids = tf.keras.layers.Input((128,), dtype=tf.int32, name='neg_ids')
    margins = tf.keras.layers.Input((), dtype=tf.float32, name='margins')
    
    anc_embs = embedder(anc_ids)
    pos_embs = embedder(pos_ids)
    neg_embs = embedder(neg_ids)
    
    inputs = [anc_ids, pos_ids, neg_ids, margins]
    outputs = [anc_embs, pos_embs, neg_embs, margins]
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def loss_fn(anc_embs, pos_embs, neg_embs, margins):
        loss = triplet_loss(anc_embs, pos_embs, neg_embs, margins)
        return tf.nn.compute_average_loss(loss, global_batch_size=batch_size)
    
    model.add_loss(loss_fn(anc_embs, pos_embs, neg_embs, margins))
    return embedder, model