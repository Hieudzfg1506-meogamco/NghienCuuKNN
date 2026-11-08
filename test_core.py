"""Kiểm tra nhanh phần dữ liệu và mô hình bằng unittest."""

from pathlib import Path
import unittest

from data_loader import load_dataset
from model import fixed_split, train_model, validate_result


ROOT = Path(__file__).resolve().parent


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset(ROOT / "archive.zip")

    def test_real_dataset(self):
        self.assertEqual(len(self.data.X), 569)
        self.assertEqual(self.data.X.shape[1], 30)
        self.assertEqual(int((self.data.y == 0).sum()), 357)
        self.assertEqual(int((self.data.y == 1).sum()), 212)
        self.assertNotIn("id", self.data.X.columns)
        self.assertNotIn("diagnosis", self.data.X.columns)
        self.assertTrue(any(name in {"Unnamed: 32", ""} for name in self.data.removed_empty_columns))

    def test_split_is_reproducible_and_stratified(self):
        train_a, test_a = fixed_split(self.data)
        train_b, test_b = fixed_split(self.data)
        self.assertListEqual(train_a.tolist(), train_b.tolist())
        self.assertListEqual(test_a.tolist(), test_b.tolist())
        self.assertEqual(len(train_a), 455)
        self.assertEqual(len(test_a), 114)

    def test_pipeline_and_invariants(self):
        result = train_model(self.data, k=5, metric="euclidean", weights="uniform")
        self.assertEqual(list(result.pipeline.named_steps), ["imputer", "scaler", "knn"])
        self.assertEqual(result.pipeline.named_steps["imputer"].statistics_.shape[0], 30)
        validate_result(result)
        self.assertEqual(int(result.matrix.sum()), 114)
        self.assertEqual(len(result.predictions), 114)


if __name__ == "__main__":
    unittest.main(verbosity=2)
