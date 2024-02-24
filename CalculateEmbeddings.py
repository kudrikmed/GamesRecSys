import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import logging
import click
from RecommendationEvaluator import RecommendationEvaluator
import mlflow
import os

logging.basicConfig(
    filename='logs/recsys.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')


class BertEmbeddingsCalculator:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
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

    def make_embeddings(self, game_df_path='data/prepared/game_df.pkl', user_df_path='data/prepared/user_df.pkl'):
        game_df = pd.read_pickle(game_df_path)
        user_df = pd.read_pickle(user_df_path)

        self.logger.info("Calculating games embeddings...")
        games_embeddings = self.calculate_bert_embeddings(game_df['CombinedColumn'])
        self.logger.info("Calculating users embeddings...")
        users_embeddings = self.calculate_bert_embeddings(user_df['CombinedColumn'])

        np.save('data/prepared/games_embeddings.npy', games_embeddings)
        np.save('data/prepared/users_embeddings.npy', users_embeddings)
        self.logger.info("Embeddings saved")

        # Logging embeddings with MLflow
        with mlflow.start_run():
            mlflow.log_params({
                'model_name': 'bert-base-uncased',
                'game_df_path': game_df_path,
                'user_df_path': user_df_path
            })

            mlflow.log_artifact(game_df_path)
            mlflow.log_artifact(user_df_path)
            if os.path.exists('rec_metrics.png'):
                mlflow.log_artifact('rec_metrics.png')
            user_id_list = ['user_' + str(i) for i in range(1, 51)]
            top_games_number = 15
            k = 5
            evaluator = RecommendationEvaluator('data/prepared/user_history_df.pkl')
            mean_metrics = evaluator.evaluate_mean_metrics_for_user_list(user_id_list, top_games_number, k)
            for key, value in mean_metrics.items():
                mlflow.log_metric(key, value)


@click.command()
@click.option('--game_df_path',
              default='data/prepared/game_df.pkl',
              help='Path to the games embeddings output file.',
              type=click.Path(exists=True))
@click.option('--user_df_path',
              default='data/prepared/user_df.pkl',
              help='Path to the users embeddings output file.',
              type=click.Path(exists=True))
def calculate_embeddings(game_df_path, user_df_path):
    embeddings_calculator = BertEmbeddingsCalculator()
    embeddings_calculator.make_embeddings(game_df_path, user_df_path)


if __name__ == '__main__':

    calculate_embeddings()
