from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CFArtifacts:
    user_index: Dict[int, int]
    item_index: Dict[int, int]
    global_mean: float
    svd_prediction_matrix: np.ndarray
    item_similarity_matrix: np.ndarray
    user_item_matrix: np.ndarray


def build_user_item_matrix(ratings: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    users = sorted(ratings["user_id"].unique().tolist())
    items = sorted(ratings["movie_id"].unique().tolist())

    user_index = {u: i for i, u in enumerate(users)}
    item_index = {m: j for j, m in enumerate(items)}

    mat = np.zeros((len(users), len(items)), dtype=np.float32)
    for row in ratings.itertuples(index=False):
        mat[user_index[int(row.user_id)], item_index[int(row.movie_id)]] = float(row.rating)

    return mat, user_index, item_index


def compute_svd_prediction_matrix(user_item_matrix: np.ndarray, k: int = 50) -> np.ndarray:
    if user_item_matrix.size == 0:
        raise ValueError("user_item_matrix is empty")

    user_means = np.where(
        (user_item_matrix != 0).sum(axis=1) > 0,
        user_item_matrix.sum(axis=1) / np.maximum((user_item_matrix != 0).sum(axis=1), 1),
        0,
    )
    centered = user_item_matrix.copy()
    non_zero_mask = centered != 0
    centered[non_zero_mask] = centered[non_zero_mask] - np.repeat(user_means, centered.shape[1])[non_zero_mask.ravel()]
    centered = centered.reshape(user_item_matrix.shape)

    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    k_eff = max(1, min(k, len(s)))
    u_k = u[:, :k_eff]
    s_k = np.diag(s[:k_eff])
    vt_k = vt[:k_eff, :]
    recon = u_k @ s_k @ vt_k
    recon = recon + user_means[:, None]
    return recon.astype(np.float32)


def compute_item_similarity(user_item_matrix: np.ndarray) -> np.ndarray:
    # Item-based CF similarity over item vectors in user-item space.
    item_vectors = user_item_matrix.T
    sim = cosine_similarity(item_vectors)
    return sim.astype(np.float32)


def fit_cf_artifacts(train_ratings: pd.DataFrame, k: int = 50) -> CFArtifacts:
    user_item_matrix, user_index, item_index = build_user_item_matrix(train_ratings)
    svd_pred = compute_svd_prediction_matrix(user_item_matrix, k=k)
    item_sim = compute_item_similarity(user_item_matrix)
    global_mean = float(train_ratings["rating"].mean())

    return CFArtifacts(
        user_index=user_index,
        item_index=item_index,
        global_mean=global_mean,
        svd_prediction_matrix=svd_pred,
        item_similarity_matrix=item_sim,
        user_item_matrix=user_item_matrix,
    )


def predict_svd_score(artifacts: CFArtifacts, user_id: int, movie_id: int) -> float:
    if user_id not in artifacts.user_index or movie_id not in artifacts.item_index:
        return artifacts.global_mean
    u = artifacts.user_index[user_id]
    i = artifacts.item_index[movie_id]
    return float(artifacts.svd_prediction_matrix[u, i])


def predict_item_based_score(artifacts: CFArtifacts, user_id: int, movie_id: int) -> float:
    if user_id not in artifacts.user_index or movie_id not in artifacts.item_index:
        return artifacts.global_mean

    u = artifacts.user_index[user_id]
    target_i = artifacts.item_index[movie_id]

    user_row = artifacts.user_item_matrix[u]
    rated_item_idx = np.where(user_row > 0)[0]
    if len(rated_item_idx) == 0:
        return artifacts.global_mean

    similarities = artifacts.item_similarity_matrix[target_i, rated_item_idx]
    ratings = user_row[rated_item_idx]

    denom = float(np.sum(np.abs(similarities)))
    if denom == 0:
        return artifacts.global_mean

    return float(np.dot(similarities, ratings) / denom)


def score_test_pairs(test_ratings: pd.DataFrame, artifacts: CFArtifacts) -> pd.DataFrame:
    rows = []
    for row in test_ratings.itertuples(index=False):
        user_id = int(row.user_id)
        movie_id = int(row.movie_id)
        rows.append(
            {
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": float(row.rating),
                "svd_score": predict_svd_score(artifacts, user_id, movie_id),
                "item_cf_score": predict_item_based_score(artifacts, user_id, movie_id),
            }
        )

    return pd.DataFrame(rows)
