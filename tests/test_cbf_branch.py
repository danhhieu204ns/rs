import unittest

import numpy as np
import pandas as pd

from src.models.cbf_branch import (
    build_item_corpus,
    fit_cbf_artifacts,
    predict_content_score,
    score_test_pairs_cbf,
)


class TestCBFBranch(unittest.TestCase):
    def setUp(self):
        self.items = pd.DataFrame(
            {
                "movie_id": [1, 2, 3],
                "title": ["Action Hero", "Romance Tale", "Hero Returns"],
                "genres_text": ["Action|Adventure", "Romance|Drama", "Action|Drama"],
            }
        )
        self.train = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 3],
                "movie_id": [1, 2, 2, 3],
                "rating": [5.0, 3.0, 4.0, 2.0],
                "timestamp": [1, 2, 3, 4],
            }
        )
        self.test = pd.DataFrame(
            {
                "user_id": [1, 2],
                "movie_id": [3, 1],
                "rating": [4.0, 3.0],
                "timestamp": [5, 6],
            }
        )

    def test_build_item_corpus(self):
        corpus = build_item_corpus(self.items)
        self.assertEqual(len(corpus), 3)
        self.assertTrue(corpus.str.len().gt(0).all())

    def test_fit_artifacts(self):
        artifacts = fit_cbf_artifacts(self.items, self.train)
        self.assertEqual(len(artifacts.movie_ids), 3)
        self.assertGreater(artifacts.tfidf_matrix.shape[1], 0)
        self.assertTrue(np.isfinite(artifacts.item_similarity_matrix).all())

    def test_predict_content_score_finite(self):
        artifacts = fit_cbf_artifacts(self.items, self.train)
        score = predict_content_score(artifacts, self.train, user_id=1, movie_id=3)
        self.assertTrue(np.isfinite(score))

    def test_score_test_pairs(self):
        artifacts = fit_cbf_artifacts(self.items, self.train)
        scored = score_test_pairs_cbf(self.test, self.train, artifacts)
        self.assertEqual(len(scored), len(self.test))
        self.assertIn("cbf_score", scored.columns)
        self.assertTrue(np.isfinite(scored["cbf_score"]).all())


if __name__ == "__main__":
    unittest.main()
