from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class CBFArtifacts:
    movie_ids: List[int]
    movie_index: Dict[int, int]
    tfidf_matrix: np.ndarray
    item_similarity_matrix: np.ndarray
    global_mean: float


def build_item_corpus(items: pd.DataFrame) -> pd.Series:
    if "genres_text" in items.columns:
        base = items["genres_text"].fillna("unknown").astype(str)
    else:
        base = pd.Series(["unknown"] * len(items), index=items.index)

    if "title" in items.columns:
        title = items["title"].fillna("").astype(str)
        corpus = (title + " " + base).str.strip()
    else:
        corpus = base

    corpus = corpus.replace("", "unknown")
    return corpus


def fit_cbf_artifacts(items: pd.DataFrame, train_ratings: pd.DataFrame) -> CBFArtifacts:
    work = items[["movie_id", "title", "genres_text"]].copy() if "genres_text" in items.columns else items[["movie_id", "title"]].copy()
    work = work.drop_duplicates(subset=["movie_id"]).sort_values("movie_id").reset_index(drop=True)

    corpus = build_item_corpus(work)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    sim_matrix = cosine_similarity(tfidf_matrix)

    movie_ids = work["movie_id"].astype(int).tolist()
    movie_index = {mid: idx for idx, mid in enumerate(movie_ids)}

    return CBFArtifacts(
        movie_ids=movie_ids,
        movie_index=movie_index,
        tfidf_matrix=tfidf_matrix.toarray().astype(np.float32),
        item_similarity_matrix=sim_matrix.astype(np.float32),
        global_mean=float(train_ratings["rating"].mean()),
    )


def predict_content_score(
    artifacts: CBFArtifacts,
    train_ratings: pd.DataFrame,
    user_id: int,
    movie_id: int,
) -> float:
    if movie_id not in artifacts.movie_index:
        return artifacts.global_mean

    user_hist = train_ratings[train_ratings["user_id"] == user_id]
    if user_hist.empty:
        return artifacts.global_mean

    target_i = artifacts.movie_index[movie_id]
    numerator = 0.0
    denominator = 0.0

    for row in user_hist.itertuples(index=False):
        hist_movie = int(row.movie_id)
        if hist_movie not in artifacts.movie_index:
            continue
        hist_i = artifacts.movie_index[hist_movie]
        sim = float(artifacts.item_similarity_matrix[target_i, hist_i])
        numerator += sim * float(row.rating)
        denominator += abs(sim)

    if denominator == 0:
        return artifacts.global_mean

    return numerator / denominator


def score_test_pairs_cbf(test_ratings: pd.DataFrame, train_ratings: pd.DataFrame, artifacts: CBFArtifacts) -> pd.DataFrame:
    rows = []
    for row in test_ratings.itertuples(index=False):
        user_id = int(row.user_id)
        movie_id = int(row.movie_id)
        rows.append(
            {
                "user_id": user_id,
                "movie_id": movie_id,
                "rating": float(row.rating),
                "cbf_score": predict_content_score(artifacts, train_ratings, user_id, movie_id),
            }
        )
    return pd.DataFrame(rows)
