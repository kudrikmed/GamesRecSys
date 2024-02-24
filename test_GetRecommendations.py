import unittest
import pandas as pd
from GetRecommendations import GameRecommendationSystem


class TestGameRecommendationSystem(unittest.TestCase):

    def setUp(self):
        self.sample_user_history_df = pd.read_pickle('data/prepared/user_history_df.pkl')
        self.sample_game_description_df = pd.read_pickle('data/prepared/game_description_df.pkl')

    def test_calculate_bert_embeddings(self):
        recommendation_system = GameRecommendationSystem()
        text = "Sample text for testing embeddings."
        embeddings = recommendation_system.calculate_bert_embeddings(pd.Series([text]))
        self.assertIsNotNone(embeddings)
        self.assertEqual(embeddings.shape, (1, 768))

    def test_get_top_similar_games(self):
        recommendation_system = GameRecommendationSystem()
        input_title = "Sample Game Title"
        top_similar_games = recommendation_system.get_top_similar_games(input_title)
        self.assertIsNotNone(top_similar_games)
        self.assertEqual(len(top_similar_games), 5)

    def test_get_top_similar_users(self):
        recommendation_system = GameRecommendationSystem()
        user_id = "user_1"
        top_similar_users = recommendation_system.get_top_similar_users(user_id)
        self.assertIsNotNone(top_similar_users)
        self.assertEqual(len(top_similar_users), 5)

    def test_get_top_rated_games_by_user(self):
        recommendation_system = GameRecommendationSystem()
        user_id = "user_1"
        top_rated_games = recommendation_system.get_top_rated_games_by_user(user_id, self.sample_user_history_df)
        self.assertIsNotNone(top_rated_games)
        self.assertEqual(len(top_rated_games), 5)

    def test_user_personal_history_recommendations(self):
        recommendation_system = GameRecommendationSystem()
        user_id = "user_1"
        recommended_games = recommendation_system.user_personal_history_recommendations(user_id)
        self.assertIsNotNone(recommended_games)

    def test_similar_users_history_recommendations(self):
        recommendation_system = GameRecommendationSystem()
        user_id = "user_1"
        recommended_games = recommendation_system.similar_users_history_recommendations(user_id)
        self.assertIsNotNone(recommended_games)
        self.assertEqual(len(recommended_games), 5)

    def test_get_combined_columns(self):
        recommendation_system = GameRecommendationSystem()
        game_titles = ["Game 1", "Game 2", "Game 3"]
        combined_columns = recommendation_system.get_combined_columns(game_titles, self.sample_game_description_df)
        self.assertIsNotNone(combined_columns)
        self.assertEqual(len(combined_columns), len(game_titles))

    def test_make_recommendations(self):
        recommendation_system = GameRecommendationSystem()
        user_id = "user_1"
        recommendations = recommendation_system.make_recommendations(user_id)
        self.assertIsNotNone(recommendations)


if __name__ == '__main__':
    unittest.main()
