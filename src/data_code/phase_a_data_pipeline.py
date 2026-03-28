from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, Tuple
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
EXPECTED_USERS = 943
EXPECTED_MOVIES = 1682
EXPECTED_RATINGS = 100000
RANDOM_STATE = 42

RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
ITEM_COLUMNS = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Childrens",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "FilmNoir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "SciFi",
    "Thriller",
    "War",
    "Western",
]
GENRE_COLUMNS = ITEM_COLUMNS[5:]


def download_and_extract_movielens_100k(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-100k.zip"
    extract_dir = raw_dir / "ml-100k"

    if extract_dir.exists() and (extract_dir / "u.data").exists() and (extract_dir / "u.item").exists():
        return extract_dir

    if not zip_path.exists():
        urlretrieve(MOVIELENS_100K_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    # Expected extraction path is raw_dir/ml-100k
    if not extract_dir.exists():
        raise FileNotFoundError("Could not find extracted ml-100k directory")

    return extract_dir


def load_movielens_100k(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        header=None,
        names=RATING_COLUMNS,
        encoding="latin-1",
    )
    items = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        header=None,
        names=ITEM_COLUMNS,
        encoding="latin-1",
    )
    return ratings, items


def validate_dataset_shape(ratings: pd.DataFrame, items: pd.DataFrame) -> Dict[str, int]:
    return {
        "users": int(ratings["user_id"].nunique()),
        "movies": int(items["movie_id"].nunique()),
        "ratings": int(len(ratings)),
    }


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    cleaned = ratings.dropna(subset=["user_id", "movie_id", "rating", "timestamp"]).copy()
    cleaned = cleaned.drop_duplicates(subset=RATING_COLUMNS)
    cleaned["user_id"] = cleaned["user_id"].astype(int)
    cleaned["movie_id"] = cleaned["movie_id"].astype(int)
    cleaned["rating"] = cleaned["rating"].astype(float)
    cleaned["timestamp"] = cleaned["timestamp"].astype(int)
    return cleaned


def clean_items(items: pd.DataFrame) -> pd.DataFrame:
    cleaned = items.dropna(subset=["movie_id", "title"]).copy()
    cleaned["movie_id"] = cleaned["movie_id"].astype(int)
    for col in GENRE_COLUMNS:
        cleaned[col] = cleaned[col].fillna(0).astype(int)
    cleaned["genres_text"] = cleaned.apply(_join_genres, axis=1)
    return cleaned


def _join_genres(row: pd.Series) -> str:
    genres = [genre for genre in GENRE_COLUMNS if int(row[genre]) == 1]
    return "|".join(genres) if genres else "unknown"


def split_train_test(ratings: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        ratings,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def has_data_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    overlap = train_df.merge(test_df, how="inner", on=RATING_COLUMNS)
    return not overlap.empty


def build_qa_report(
    raw_shape: Dict[str, int],
    clean_shape: Dict[str, int],
    train_size: int,
    test_size: int,
    leakage: bool,
) -> Dict[str, object]:
    return {
        "raw_shape": raw_shape,
        "clean_shape": clean_shape,
        "expected": {
            "users": EXPECTED_USERS,
            "movies": EXPECTED_MOVIES,
            "ratings": EXPECTED_RATINGS,
            "test_ratio": 0.2,
        },
        "checks": {
            "users_match_expected": clean_shape["users"] == EXPECTED_USERS,
            "movies_match_expected": clean_shape["movies"] == EXPECTED_MOVIES,
            "ratings_match_expected": clean_shape["ratings"] == EXPECTED_RATINGS,
            "no_leakage": not leakage,
        },
        "split": {
            "train_size": train_size,
            "test_size": test_size,
            "test_ratio": test_size / (train_size + test_size),
        },
    }


def run_phase_a(raw_dir: Path, processed_dir: Path, reports_dir: Path) -> Dict[str, object]:
    data_dir = download_and_extract_movielens_100k(raw_dir)
    ratings_raw, items_raw = load_movielens_100k(data_dir)

    raw_shape = validate_dataset_shape(ratings_raw, items_raw)
    ratings_clean = clean_ratings(ratings_raw)
    items_clean = clean_items(items_raw)
    clean_shape = validate_dataset_shape(ratings_clean, items_clean)

    train_df, test_df = split_train_test(ratings_clean, test_size=0.2)
    leakage = has_data_leakage(train_df, test_df)

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    ratings_clean.to_csv(processed_dir / "ratings_clean.csv", index=False)
    items_clean.to_csv(processed_dir / "items_clean.csv", index=False)
    train_df.to_csv(processed_dir / "ratings_train.csv", index=False)
    test_df.to_csv(processed_dir / "ratings_test.csv", index=False)

    split_indices = {
        "train_index": train_df.index.tolist(),
        "test_index": test_df.index.tolist(),
    }
    with (processed_dir / "split_indices.json").open("w", encoding="utf-8") as f:
        json.dump(split_indices, f)

    report = build_qa_report(
        raw_shape=raw_shape,
        clean_shape=clean_shape,
        train_size=len(train_df),
        test_size=len(test_df),
        leakage=leakage,
    )
    with (reports_dir / "phase_a_qa_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase A data pipeline for strict HRS-IU-DL replication")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_phase_a(args.raw_dir, args.processed_dir, args.reports_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
