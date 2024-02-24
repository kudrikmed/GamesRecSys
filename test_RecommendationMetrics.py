import unittest
from RecommendationMetrics import RecommendationMetrics


class TestRecommendationMetrics(unittest.TestCase):

    def test_precision_at_k(self):
        actual = [1, 2, 3, 4, 5]
        predicted = [1, 3, 5, 7, 9]
        k = 3

        precision = RecommendationMetrics.precision_at_k(actual, predicted, k)
        self.assertAlmostEqual(precision, 2 / 3, places=2)

    def test_recall_at_k(self):
        actual = [1, 2, 3, 4, 5]
        predicted = [1, 3, 5, 7, 9]
        k = 3

        recall = RecommendationMetrics.recall_at_k(actual, predicted, k)
        self.assertAlmostEqual(recall, 2 / min(k, len(actual)), places=2)

    def test_mean_reciprocal_rank(self):
        actual = [1, 2, 3, 4, 5]
        predicted = [1, 3, 5, 7, 9]

        mrr = RecommendationMetrics.mean_reciprocal_rank(actual, predicted)
        self.assertAlmostEqual(mrr, 1 / 1, places=2)

    def test_normalized_discounted_cumulative_gain(self):
        actual = [1, 2, 3, 4, 5]
        predicted = [1, 3, 5, 7, 9]
        k = 3

        ndcg = RecommendationMetrics.normalized_discounted_cumulative_gain(actual, predicted, k)
        self.assertAlmostEqual(ndcg, (1 / 1 + 1 / 2 + 1 / 3) / (1 / 1 + 1 / 2 + 1 / 3), places=2)


if __name__ == '__main__':
    unittest.main()
