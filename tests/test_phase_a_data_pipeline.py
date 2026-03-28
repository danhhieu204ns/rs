import unittest

import pandas as pd

from src.data.phase_a_data_pipeline import (
    RATING_COLUMNS,
    build_qa_report,
    clean_items,
    clean_ratings,
    has_data_leakage,
    split_train_test,
    validate_dataset_shape,
)


class TestPhaseADataPipeline(unittest.TestCase):
    def test_clean_ratings_drops_null_and_duplicates(self):
        ratings = pd.DataFrame(
            [
                [1, 1, 5, 100],
                [1, 1, 5, 100],
                [2, 3, 4, 101],
                [3, None, 3, 102],
            ],
            columns=RATING_COLUMNS,
        )
        cleaned = clean_ratings(ratings)
        self.assertEqual(len(cleaned), 2)
        self.assertTrue(cleaned["movie_id"].notna().all())

    def test_split_ratio_and_no_leakage(self):
        ratings = pd.DataFrame(
            {
                "user_id": list(range(1, 101)),
                "movie_id": list(range(1, 101)),
                "rating": [4.0] * 100,
                "timestamp": list(range(1000, 1100)),
            }
        )
        train_df, test_df = split_train_test(ratings, test_size=0.2)
        self.assertEqual(len(train_df), 80)
        self.assertEqual(len(test_df), 20)
        self.assertFalse(has_data_leakage(train_df, test_df))

    def test_clean_items_adds_genres_text(self):
        items = pd.DataFrame(
            [
                [1, "Movie A", "01-Jan-1995", None, "url", 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, "Movie B", "01-Jan-1996", None, "url", 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            columns=[
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
            ],
        )
        cleaned = clean_items(items)
        self.assertIn("genres_text", cleaned.columns)
        self.assertTrue(cleaned["genres_text"].str.len().gt(0).all())

    def test_build_qa_report_structure(self):
        raw_shape = {"users": 943, "movies": 1682, "ratings": 100000}
        clean_shape = {"users": 943, "movies": 1682, "ratings": 100000}
        report = build_qa_report(raw_shape, clean_shape, train_size=80000, test_size=20000, leakage=False)
        self.assertTrue(report["checks"]["users_match_expected"])
        self.assertTrue(report["checks"]["movies_match_expected"])
        self.assertTrue(report["checks"]["ratings_match_expected"])
        self.assertTrue(report["checks"]["no_leakage"])

    def test_validate_dataset_shape(self):
        ratings = pd.DataFrame(
            {
                "user_id": [1, 1, 2],
                "movie_id": [1, 2, 1],
                "rating": [5, 4, 3],
                "timestamp": [1, 2, 3],
            }
        )
        items = pd.DataFrame(
            {
                "movie_id": [1, 2],
                "title": ["A", "B"],
            }
        )
        shape = validate_dataset_shape(ratings, items)
        self.assertEqual(shape["users"], 2)
        self.assertEqual(shape["movies"], 2)
        self.assertEqual(shape["ratings"], 3)


if __name__ == "__main__":
    unittest.main()
