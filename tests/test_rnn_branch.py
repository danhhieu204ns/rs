import unittest
import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.rnn_branch import (
    RNNArtifacts,
    build_index_maps,
    encode_rnn_data,
    build_rnn_model,
    fit_rnn,
    predict_rnn_score,
    score_test_pairs_rnn
)


class TestRNNBranch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass

        cls.train_df = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2],
                "movie_id": [10, 20, 30, 10, 40],
                "rating": [5.0, 4.0, 3.0, 4.0, 5.0],
                "timestamp": [100, 101, 102, 103, 104],
            }
        )
        cls.test_df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "movie_id": [40, 30, 50],
                "rating": [4.0, 4.0, 3.0],
                "timestamp": [105, 106, 107],
            }
        )

    def test_build_index_maps(self):
        user_idx, item_idx = build_index_maps(self.train_df)
        self.assertEqual(len(user_idx), 2)
        self.assertEqual(len(item_idx), 4) # 10, 20, 30, 40
        self.assertNotIn(0, item_idx.values()) # 0 is reserved

    def test_encode_rnn_data(self):
        user_idx, item_idx = build_index_maps(self.train_df)
        seqs, items, ratings, hists = encode_rnn_data(self.train_df, user_idx, item_idx, max_seq_len=2)
        
        self.assertEqual(seqs.shape, (5, 2))
        self.assertEqual(items.shape, (5,))
        self.assertEqual(ratings.shape, (5,))
        
        # User 1 history should be length 3, last 2 items padding to right logic?
        # Encodings check: user 1 sees 10, 20, 30.
        # hists[user_idx[1]] should end up being [item_idx[10], item_idx[20], item_idx[30]]
        self.assertEqual(len(hists[user_idx[1]]), 3)

    def test_fit_and_score(self):
        # Very small dataset, 1 epoch just to test forward pass
        artifacts = fit_rnn(
            self.train_df,
            embedding_dim=4,
            max_seq_len=2,
            epochs=1,
            batch_size=2
        )
        
        self.assertIsInstance(artifacts, RNNArtifacts)
        
        # Predict 
        scored_df = score_test_pairs_rnn(self.test_df, artifacts)
        
        self.assertEqual(len(scored_df), 3)
        self.assertIn("rnn_score", scored_df.columns)
        
        # New user (3) or item (50) should get global mean
        new_user_score = scored_df.loc[scored_df["user_id"] == 3, "rnn_score"].values[0]
        self.assertAlmostEqual(new_user_score, self.train_df["rating"].mean())
        
        # Ensure finite scores
        self.assertTrue(np.isfinite(scored_df["rnn_score"]).all())


if __name__ == "__main__":
    unittest.main()
