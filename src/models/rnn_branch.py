from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


def configure_tf_runtime() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


@dataclass
class RNNArtifacts:
    model: tf.keras.Model
    user_index: Dict[int, int]
    item_index: Dict[int, int]
    user_histories: Dict[int, list[int]]
    global_mean: float
    max_seq_len: int


def build_index_maps(train_ratings: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    users = sorted(train_ratings["user_id"].unique().tolist())
    items = sorted(train_ratings["movie_id"].unique().tolist())
    user_index = {u: i for i, u in enumerate(users)}
    item_index = {m: i + 1 for i, m in enumerate(items)}  # 0 reserved for padding
    return user_index, item_index


def encode_rnn_data(
    df: pd.DataFrame, 
    user_index: Dict[int, int], 
    item_index: Dict[int, int], 
    max_seq_len: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, list[int]]]:
    # Sort by time to create chronological sequences
    if "timestamp" in df.columns:
        df = df.sort_values(by=["user_id", "timestamp"])
    
    seqs = []
    items = []
    ratings = []
    
    user_histories: Dict[int, list[int]] = {u: [] for u in user_index.values()}
    
    for row in df.itertuples(index=False):
        u = int(row.user_id)
        m = int(row.movie_id)
        if u not in user_index or m not in item_index:
            continue
            
        uidx = user_index[u]
        midx = item_index[m]
        
        # Build sequence from past interactions up to max_seq_len
        history = user_histories[uidx][-max_seq_len:]
        padded_seq = history + [0] * (max_seq_len - len(history))
        
        seqs.append(padded_seq)
        items.append(midx)
        ratings.append(float(row.rating))
        
        # Update user's history with the current interaction
        user_histories[uidx].append(midx)
        
    return (
        np.asarray(seqs, dtype=np.int32),
        np.asarray(items, dtype=np.int32),
        np.asarray(ratings, dtype=np.float32),
        user_histories
    )


def build_rnn_model(n_items_with_pad: int, max_seq_len: int, embedding_dim: int = 32) -> tf.keras.Model:
    seq_in = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name="history_seq")
    item_in = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="target_item")
    
    item_emb_layer = tf.keras.layers.Embedding(
        n_items_with_pad, 
        embedding_dim, 
        mask_zero=True, 
        name="item_embedding"
    )
    
    seq_emb = item_emb_layer(seq_in)
    target_emb = item_emb_layer(item_in)
    
    # Process sequential user history
    rnn_out = tf.keras.layers.SimpleRNN(embedding_dim, name="rnn_layer")(seq_emb)
    
    # The dot product of user state and target item state predicts the rating
    dot = tf.keras.layers.Dot(axes=-1, name="dot_product")([rnn_out, target_emb])
    out = tf.keras.layers.Flatten(name="flatten_output")(dot)
    
    model = tf.keras.Model(inputs=[seq_in, item_in], outputs=out, name="rnn_model")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def fit_rnn(
    train_ratings: pd.DataFrame,
    embedding_dim: int = 32,
    max_seq_len: int = 20,
    epochs: int = 5,
    batch_size: int = 512,
) -> RNNArtifacts:
    configure_tf_runtime()
    tf.keras.utils.set_random_seed(42)

    user_index, item_index = build_index_maps(train_ratings)
    X_seq, X_item, y, user_histories = encode_rnn_data(train_ratings, user_index, item_index, max_seq_len)

    model = build_rnn_model(len(item_index) + 1, max_seq_len, embedding_dim=embedding_dim)
    model.fit([X_seq, X_item], y, epochs=epochs, batch_size=batch_size, verbose=0)

    return RNNArtifacts(
        model=model,
        user_index=user_index,
        item_index=item_index,
        user_histories=user_histories,
        global_mean=float(train_ratings["rating"].mean()),
        max_seq_len=max_seq_len,
    )


def predict_rnn_score(artifacts: RNNArtifacts, user_id: int, movie_id: int) -> float:
    # Handle Cold-start with global mean
    if user_id not in artifacts.user_index or movie_id not in artifacts.item_index:
        return artifacts.global_mean

    uidx = artifacts.user_index[user_id]
    midx = artifacts.item_index[movie_id]
    
    history = artifacts.user_histories[uidx][-artifacts.max_seq_len:]
    padded_seq = history + [0] * (artifacts.max_seq_len - len(history))
    
    s = np.asarray([padded_seq], dtype=np.int32)
    i = np.asarray([midx], dtype=np.int32)
    
    pred = artifacts.model.predict([s, i], verbose=0)
    return float(pred[0][0])


def score_test_pairs_rnn(test_ratings: pd.DataFrame, artifacts: RNNArtifacts) -> pd.DataFrame:
    scored = test_ratings[["user_id", "movie_id", "rating"]].copy()
    scored["rnn_score"] = artifacts.global_mean

    known_mask = scored["user_id"].isin(artifacts.user_index) & scored["movie_id"].isin(artifacts.item_index)
    if not known_mask.any():
        return scored

    known = scored.loc[known_mask, ["user_id", "movie_id"]].copy()
    known["uidx"] = known["user_id"].map(artifacts.user_index)
    known["midx"] = known["movie_id"].map(artifacts.item_index)

    seq_cache: Dict[int, np.ndarray] = {}
    for uidx in known["uidx"].drop_duplicates().tolist():
        history = artifacts.user_histories.get(uidx, [])[-artifacts.max_seq_len:]
        padded_seq = history + [0] * (artifacts.max_seq_len - len(history))
        seq_cache[int(uidx)] = np.asarray(padded_seq, dtype=np.int32)

    seq_batch = np.stack([seq_cache[int(uidx)] for uidx in known["uidx"].tolist()], axis=0)
    item_batch = known["midx"].to_numpy(dtype=np.int32)

    preds = artifacts.model.predict([seq_batch, item_batch], verbose=0, batch_size=2048).reshape(-1)
    scored.loc[known_mask, "rnn_score"] = preds.astype(np.float32)
    return scored
