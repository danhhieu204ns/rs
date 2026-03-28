import unittest

import numpy as np
import pandas as pd

from src.models.cf_branch import (
    compute_item_similarity,
    fit_cf_artifacts,
    predict_item_based_score,
    predict_svd_score,
    score_test_pairs,
)


class TestCFBranch(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3, 3],
                "movie_id": [1, 2, 1, 3, 2, 3],
                "rating": [5.0, 4.0, 4.0, 2.0, 3.0, 5.0],
                "timestamp": [1, 2, 3, 4, 5, 6],
            }
        )
        self.test = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "movie_id": [3, 2, 1],
                "rating": [3.0, 4.0, 2.0],
                "timestamp": [7, 8, 9],
            }
        )

    def test_fit_artifacts(self):
        artifacts = fit_cf_artifacts(self.train, k=2)
        self.assertGreater(artifacts.svd_prediction_matrix.shape[0], 0)
        self.assertTrue(np.isfinite(artifacts.svd_prediction_matrix).all())

    def test_item_similarity_diagonal_is_one(self):
        artifacts = fit_cf_artifacts(self.train, k=2)
        diag = np.diag(artifacts.item_similarity_matrix)
        self.assertTrue(np.allclose(diag, np.ones_like(diag)))

    def test_predict_scores_are_finite(self):
        artifacts = fit_cf_artifacts(self.train, k=2)
        s1 = predict_svd_score(artifacts, 1, 1)
        s2 = predict_item_based_score(artifacts, 1, 1)
        self.assertTrue(np.isfinite(s1))
        self.assertTrue(np.isfinite(s2))

    def test_score_test_pairs_shape(self):
        artifacts = fit_cf_artifacts(self.train, k=2)
        scored = score_test_pairs(self.test, artifacts)
        self.assertEqual(len(scored), len(self.test))
        self.assertIn("svd_score", scored.columns)
        self.assertIn("item_cf_score", scored.columns)

    def test_compute_item_similarity(self):
        mat = np.array([[5, 0, 1], [4, 2, 0]], dtype=np.float32)
        sim = compute_item_similarity(mat)
        self.assertEqual(sim.shape, (3, 3))
        self.assertTrue(np.allclose(np.diag(sim), np.ones(3)))


if __name__ == "__main__":
    unittest.main()
