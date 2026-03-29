import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

from src.models.phase_f_fusion_run import (
    load_and_merge_scores,
    compute_weighted_fusion,
    calculate_metrics
)


class TestPhaseFFusion(unittest.TestCase):
    def setUp(self):
        # Create mock dataframes for each scoring phase
        self.df_cf = pd.DataFrame({
            "user_id": [1, 2],
            "movie_id": [10, 20],
            "rating": [5.0, 3.0],
            "svd_score": [4.5, 3.2],
            "item_cf_score": [4.6, 2.9]
        })
        
        self.df_cbf = pd.DataFrame({
            "user_id": [1, 2],
            "movie_id": [10, 20],
            "rating": [5.0, 3.0],
            "cbf_score": [4.0, 3.5]
        })
        
        self.df_ncf = pd.DataFrame({
            "user_id": [1, 2],
            "movie_id": [10, 20],
            "rating": [5.0, 3.0],
            "ncf_score": [4.8, 3.1]
        })
        
        self.df_rnn = pd.DataFrame({
            "user_id": [1, 2],
            "movie_id": [10, 20],
            "rating": [5.0, 3.0],
            "rnn_score": [4.7, 3.0]
        })

    def test_merge_and_fusion(self):
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td)
            # Save mocks
            p_cf = base_dir / "cf.csv"
            p_cbf = base_dir / "cbf.csv"
            p_ncf = base_dir / "ncf.csv"
            p_rnn = base_dir / "rnn.csv"
            
            self.df_cf.to_csv(p_cf, index=False)
            self.df_cbf.to_csv(p_cbf, index=False)
            self.df_ncf.to_csv(p_ncf, index=False)
            self.df_rnn.to_csv(p_rnn, index=False)
            
            paths = {
                "CF": p_cf,
                "CBF": p_cbf,
                "NCF": p_ncf,
                "RNN": p_rnn
            }
            
            # Test merging
            df_merged = load_and_merge_scores(paths)
            self.assertEqual(len(df_merged), 2)
            expected_cols = {
                "user_id", "movie_id", "rating", 
                "svd_score", "item_cf_score", "cbf_score", 
                "ncf_score", "rnn_score"
            }
            self.assertEqual(set(df_merged.columns), expected_cols)
            
            # Test fusion weights
            df_fusion = compute_weighted_fusion(
                df_merged, alpha=0.2, beta=0.2, gamma=0.2, delta=0.2, epsilon=0.2
            )
            self.assertIn("final_score", df_fusion.columns)
            
            # Manual calculation for user 1: 0.2*(4.5 + 4.6 + 4.0 + 4.8 + 4.7) = 0.2*(22.6) = 4.52
            expected_val = 0.2 * (4.5 + 4.6 + 4.0 + 4.8 + 4.7)
            np.testing.assert_almost_equal(df_fusion.loc[0, "final_score"], expected_val)

    def test_metrics(self):
        df_fused = pd.DataFrame({
            "rating": [5.0, 3.0, 4.0, 2.0],
            "final_score": [4.0, 3.5, 3.8, 2.5]
        })
        
        metrics = calculate_metrics(df_fused)
        self.assertIn("RMSE", metrics)
        self.assertIn("MAE", metrics)
        self.assertIn("Precision", metrics)
        self.assertIn("Recall", metrics)
        
        # Check finite values
        self.assertTrue(np.isfinite(metrics["RMSE"]))
        self.assertTrue(metrics["RMSE"] > 0)

if __name__ == "__main__":
    unittest.main()
