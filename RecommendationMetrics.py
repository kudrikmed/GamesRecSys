class RecommendationMetrics:
    @staticmethod
    def precision_at_k(actual, predicted, k):
        if k <= 0:
            raise ValueError("K should be a positive integer.")

        actual_set = set(actual[:k])
        predicted_set = set(predicted[:k])

        intersection = actual_set.intersection(predicted_set)
        precision = len(intersection) / k if k > 0 else 0.0

        return precision

    @staticmethod
    def recall_at_k(actual, predicted, k):
        if k <= 0:
            raise ValueError("K should be a positive integer.")

        actual_set = set(actual[:k])
        predicted_set = set(predicted[:k])

        intersection = actual_set.intersection(predicted_set)
        recall = len(intersection) / len(actual_set) if len(actual_set) > 0 else 0.0

        return recall

    @staticmethod
    def mean_average_precision(actual, predicted):
        total_precision = 0.0
        relevant_count = 0

        for i, pred_item in enumerate(predicted):
            if pred_item in actual:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                total_precision += precision_at_i

        if relevant_count == 0:
            return 0.0

        map_score = total_precision / relevant_count
        return map_score

    @staticmethod
    def mean_reciprocal_rank(actual, predicted):
        for i, pred_item in enumerate(predicted):
            if pred_item in actual:
                return 1.0 / (i + 1)

        return 0.0

    @staticmethod
    def normalized_discounted_cumulative_gain(actual, predicted, k):
        if k <= 0:
            raise ValueError("K should be a positive integer.")

        actual_set = set(actual)
        dcg = 0.0

        for i in range(k):
            item = predicted[i] if i < len(predicted) else None
            if item in actual_set:
                relevance = 1 / (i + 1)
                dcg += relevance

        ideal_order = sorted(actual, key=lambda x: actual.index(x))
        idcg = sum(1 / (i + 1) for i in range(min(k, len(ideal_order))))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        return ndcg
