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
            # Some backends may not support memory growth; continue with defaults.
            pass


@dataclass
class NCFArtifacts:
    model: tf.keras.Model
    user_index: Dict[int, int]
    item_index: Dict[int, int]
    global_mean: float


def build_index_maps(train_ratings: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    users = sorted(train_ratings["user_id"].unique().tolist())
    items = sorted(train_ratings["movie_id"].unique().tolist())
    user_index = {u: i for i, u in enumerate(users)}
    item_index = {m: i for i, m in enumerate(items)}
    return user_index, item_index


def encode_pairs(df: pd.DataFrame, user_index: Dict[int, int], item_index: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_ids = []
    item_ids = []
    ratings = []
    for row in df.itertuples(index=False):
        u = int(row.user_id)
        m = int(row.movie_id)
        if u not in user_index or m not in item_index:
            continue
        user_ids.append(user_index[u])
        item_ids.append(item_index[m])
        ratings.append(float(row.rating))

    return (
        np.asarray(user_ids, dtype=np.int32),
        np.asarray(item_ids, dtype=np.int32),
        np.asarray(ratings, dtype=np.float32),
    )


def build_ncf_model(n_users: int, n_items: int, embedding_dim: int = 32) -> tf.keras.Model:
    user_in = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="user_id")
    item_in = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="item_id")

    user_emb = tf.keras.layers.Embedding(n_users, embedding_dim, name="user_embedding")(user_in)
    item_emb = tf.keras.layers.Embedding(n_items, embedding_dim, name="item_embedding")(item_in)

    dot = tf.keras.layers.Dot(axes=-1, name="dot_product")([user_emb, item_emb])
    out = tf.keras.layers.Flatten(name="flatten_output")(dot)

    model = tf.keras.Model(inputs=[user_in, item_in], outputs=out, name="ncf_dot_model")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def fit_ncf(
    train_ratings: pd.DataFrame,
    embedding_dim: int = 32,
    epochs: int = 5,
    batch_size: int = 512,
) -> NCFArtifacts:
    configure_tf_runtime()
    tf.keras.utils.set_random_seed(42)

    user_index, item_index = build_index_maps(train_ratings)
    x_user, x_item, y = encode_pairs(train_ratings, user_index, item_index)

    model = build_ncf_model(len(user_index), len(item_index), embedding_dim=embedding_dim)
    model.fit([x_user, x_item], y, epochs=epochs, batch_size=batch_size, verbose=0)

    return NCFArtifacts(
        model=model,
        user_index=user_index,
        item_index=item_index,
        global_mean=float(train_ratings["rating"].mean()),
    )


def predict_ncf_score(artifacts: NCFArtifacts, user_id: int, movie_id: int) -> float:
    if user_id not in artifacts.user_index or movie_id not in artifacts.item_index:
        return artifacts.global_mean

    u = np.asarray([artifacts.user_index[user_id]], dtype=np.int32)
    i = np.asarray([artifacts.item_index[movie_id]], dtype=np.int32)
    pred = artifacts.model.predict([u, i], verbose=0)
    return float(pred[0][0])


def score_test_pairs_ncf(test_ratings: pd.DataFrame, artifacts: NCFArtifacts) -> pd.DataFrame:
    scored = test_ratings[["user_id", "movie_id", "rating"]].copy()
    scored["ncf_score"] = artifacts.global_mean

    known_mask = scored["user_id"].isin(artifacts.user_index) & scored["movie_id"].isin(artifacts.item_index)
    if known_mask.any():
        known = scored.loc[known_mask, ["user_id", "movie_id"]]
        x_user = known["user_id"].map(artifacts.user_index).to_numpy(dtype=np.int32)
        x_item = known["movie_id"].map(artifacts.item_index).to_numpy(dtype=np.int32)

        preds = artifacts.model.predict([x_user, x_item], verbose=0, batch_size=4096).reshape(-1)
        scored.loc[known_mask, "ncf_score"] = preds.astype(np.float32)

    return scored
