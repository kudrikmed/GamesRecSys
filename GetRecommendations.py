import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import logging
import click

logging.basicConfig(
    filename='logs/recsys.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')


class GameRecommendationSystem:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.games_embeddings = np.load('data/prepared/games_embeddings.npy')
        self.users_embeddings = np.load('data/prepared/users_embeddings.npy')
        self.game_df = pd.read_pickle('data/prepared/game_df.pkl')
        self.user_df = pd.read_pickle('data/prepared/user_df.pkl')
        self.user_history_df = pd.read_pickle('data/prepared/user_history_df.pkl')
        self.mean_ratings = pd.read_pickle('data/prepared/mean_ratings.pkl')
        self.game_description_df = pd.read_pickle('data/prepared/game_description_df.pkl')
        self.logger = logging.getLogger(__name__)

    def calculate_bert_embeddings(self, texts):
        tokenized = texts.apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True))
        max_len = max(map(len, tokenized))
        padded = [i + [0]*(max_len-len(i)) for i in tokenized]
        input_ids = torch.tensor(padded)

        with torch.no_grad():
            outputs = self.model(input_ids)

        bert_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return bert_embeddings

    def get_top_similar_games(self, input_title):
        input_embedding = self.calculate_bert_embeddings(pd.Series([input_title]))
        cosine_sim = cosine_similarity(input_embedding, self.games_embeddings)[0]
        similarity_df = pd.DataFrame({'GameTitle': self.game_df['GameTitle'], 'Similarity': cosine_sim})
        similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)
        top_similar_games = similarity_df.loc[similarity_df['GameTitle'] != input_title].head(5)

        return top_similar_games

    def get_top_similar_users(self, user_id):
        input_embedding = self.calculate_bert_embeddings(pd.Series([user_id]))
        cosine_sim = cosine_similarity(input_embedding, self.users_embeddings)[0]
        similarity_df = pd.DataFrame({'UserID': self.user_df['UserID'], 'Similarity': cosine_sim})
        similarity_df = similarity_df.sort_values(by='Similarity', ascending=False)
        top_similar_users = similarity_df.loc[similarity_df['UserID'] != user_id].head(5)

        return top_similar_users

    def get_top_rated_games_by_user(self, user_id, dataframe):
        user_data = dataframe[dataframe['UserID'] == user_id]
        sorted_data = user_data.sort_values(by='Rating', ascending=False)
        top_games = sorted_data.head(5)['GameTitle'].tolist()

        return top_games

    def user_personal_history_recommendations(self, user_id):
        top_rated_games_by_user = self.get_top_rated_games_by_user(user_id, self.user_history_df)

        games_list = []

        for game in top_rated_games_by_user:
            games_list.append(self.get_top_similar_games(game))

        unique_game_titles = set()

        for df in games_list:
            unique_game_titles.update(df['GameTitle'])

        return list(unique_game_titles)

    def similar_users_history_recommendations(self, input_user_id):
        top_similar_users = self.get_top_similar_users(input_user_id)
        user_ids = top_similar_users['UserID'].tolist()

        games_list = []
        for user_id in user_ids:
            top_games = self.get_top_rated_games_by_user(user_id, self.user_history_df)
            games_list += top_games

        filtered_df = self.mean_ratings[self.mean_ratings['GameTitle'].isin(games_list)]
        top_games = filtered_df.nlargest(5, 'Rating')
        game_titles_list = top_games['GameTitle'].tolist()

        return game_titles_list

    def get_combined_columns(self, game_titles, dataframe):
        result = []
        for game_title in game_titles:
            row = dataframe[dataframe['GameTitle'] == game_title]
            if not row.empty:
                result.append(row['CombinedColumn'].values[0])
            else:
                result.append(None)
        return result

    def make_recommendations(self, input_user_id):
        try:
            self.logger.info("Calculating recommendations for user %s...", input_user_id)

            top_rated_games_by_user = self.get_top_rated_games_by_user(input_user_id, self.user_history_df)
            my_history = self.get_combined_columns(top_rated_games_by_user, self.game_description_df)

            similar_games_list = self.similar_users_history_recommendations(input_user_id)
            similar_games = self.get_combined_columns(similar_games_list, self.game_description_df)

            similar_users_history_list = self.user_personal_history_recommendations(input_user_id)
            similar_users_history = self.get_combined_columns(similar_users_history_list, self.game_description_df)

            recommendations = {"my_history": my_history, "similar_games": similar_games,
                               "similar_users_history": similar_users_history}
            self.logger.info("Recommendations calculated successfully.")
            return recommendations

        except Exception as e:
            self.logger.error("Error during recommendation calculation: %s", str(e))


@click.command()
@click.argument('user_id', default='user_1')
def generate_recommendations(user_id):
    logging.basicConfig(level=logging.INFO)
    recommendation_system = GameRecommendationSystem()
    recommendation_system.make_recommendations(user_id)


if __name__ == '__main__':
    generate_recommendations()
