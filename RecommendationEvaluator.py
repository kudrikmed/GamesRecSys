import pandas as pd
import matplotlib.pyplot as plt
from RecommendationMetrics import RecommendationMetrics
from GetRecommendations import GameRecommendationSystem


class RecommendationEvaluator:
    def __init__(self, user_history_pickle_path):
        self.user_history_df = pd.read_pickle(user_history_pickle_path)

    def filter_and_sort_by_rating(self, user_id):
        user_data = self.user_history_df[self.user_history_df['UserID'] == user_id]
        sorted_user_data = user_data.sort_values(by='Rating', ascending=False)
        return sorted_user_data

    @staticmethod
    def get_top_k_game_titles(dataframe, k):
        top_k_titles = dataframe['GameTitle'].head(k).tolist()
        return top_k_titles

    @staticmethod
    def evaluate_recommendation_metrics(actual, predicted, k):
        metrics = RecommendationMetrics()

        precision = metrics.precision_at_k(actual, predicted, k)
        recall = metrics.recall_at_k(actual, predicted, k)
        map_score = metrics.mean_average_precision(actual, predicted)
        mrr_score = metrics.mean_reciprocal_rank(actual, predicted)
        ndcg_score = metrics.normalized_discounted_cumulative_gain(actual, predicted, k)

        return precision, recall, map_score, mrr_score, ndcg_score

    def evaluate_mean_metrics_for_user_list(self, user_id_list, top_games_number, k):
        metrics_sum = {metric: 0.0 for metric in ["precision_at_k_users",
                                                  "recall_at_k_users",
                                                  "map_score_users",
                                                  "mrr_score_users",
                                                  "ndcg_score_users",
                                                  "precision_at_k_items",
                                                  "recall_at_k_items",
                                                  "map_score_items",
                                                  "mrr_score_items",
                                                  "ndcg_score_items"]}

        for user_id in user_id_list:
            result_df = self.filter_and_sort_by_rating(user_id)
            top_k_games = self.get_top_k_game_titles(result_df, top_games_number)

            game_recommender = GameRecommendationSystem()
            recommendations = game_recommender.make_recommendations(user_id)

            games_list_similar_games = [i.split()[0] for i in recommendations['similar_games']]
            games_list_similar_users = [i.split()[0] for i in recommendations['similar_users_history']]

            top_k_games = top_k_games[5:]

            precision_at_k_users, recall_at_k_users, map_score_users, mrr_score_users, ndcg_score_users = self.evaluate_recommendation_metrics(top_k_games, games_list_similar_users, k)
            precision_at_k_items, recall_at_k_items, map_score_items, mrr_score_items, ndcg_score_items = self.evaluate_recommendation_metrics(top_k_games, games_list_similar_games, k)

            metrics_sum["precision_at_k_users"] += precision_at_k_users
            metrics_sum["recall_at_k_users"] += recall_at_k_users
            metrics_sum["map_score_users"] += map_score_users
            metrics_sum["mrr_score_users"] += mrr_score_users
            metrics_sum["ndcg_score_users"] += ndcg_score_users
            metrics_sum["precision_at_k_items"] += precision_at_k_items
            metrics_sum["recall_at_k_items"] += recall_at_k_items
            metrics_sum["map_score_items"] += map_score_items
            metrics_sum["mrr_score_items"] += mrr_score_items
            metrics_sum["ndcg_score_items"] += ndcg_score_items

        num_users = len(user_id_list)
        mean_metrics = {metric: metrics_sum[metric] / num_users for metric in metrics_sum}
        self.visualize_mean_metrics(mean_metrics)
        return mean_metrics

    @staticmethod
    def visualize_mean_metrics(mean_metrics):
        metrics_labels = list(mean_metrics.keys())
        metrics_values = list(mean_metrics.values())

        plt.bar(metrics_labels, metrics_values)
        plt.title('Mean Recommendation Metrics')
        plt.xlabel('Metrics')
        plt.xticks(rotation=45, ha="right")
        plt.ylabel('Mean Value')
        plt.tight_layout()
        plt.savefig('rec_metrics.png')


def evaluate_metrics():
    user_id_list = ['user_' + str(i) for i in range(1, 51)]
    top_games_number = 15
    k = 5
    evaluator = RecommendationEvaluator('data/prepared/user_history_df.pkl')
    mean_metrics = evaluator.evaluate_mean_metrics_for_user_list(user_id_list, top_games_number, k)
    evaluator.visualize_mean_metrics(mean_metrics)


if __name__ == '__main__':
    evaluate_metrics()
