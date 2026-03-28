import unittest

import numpy as np
import pandas as pd

from src.models.ncf_branch import fit_ncf, score_test_pairs_ncf


class TestNCFBranch(unittest.TestCase):
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

    def test_fit_and_score(self):
        artifacts = fit_ncf(self.train, embedding_dim=8, epochs=1, batch_size=2)
        scored = score_test_pairs_ncf(self.test, artifacts)
        self.assertEqual(len(scored), len(self.test))
        self.assertIn("ncf_score", scored.columns)
        self.assertTrue(np.isfinite(scored["ncf_score"]).all())


if __name__ == "__main__":
    unittest.main()
