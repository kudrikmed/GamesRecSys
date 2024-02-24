import pandas as pd
import logging
import click

logging.basicConfig(
    filename='logs/recsys.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, file_path):
        self.df = pd.read_parquet(file_path)

    def process_user_data(self):
        logger.info("Processing users data...")
        user_columns = ['UserID', 'Age', 'Gender', 'Location', 'Device', 'PlayTimeOfDay', 'TotalPlaytimeInHours', 'PurchaseHistory', 'InvolvementLevel', 'UserReview']
        user_df = self.df[user_columns].drop_duplicates().reset_index(drop=True)
        user_df['CombinedColumn'] = user_df.apply(lambda row: ' '.join(map(str, row)), axis=1)
        user_df['CombinedColumn'] = user_df['CombinedColumn'].str.lower()
        user_df.drop(['Age', 'Gender', 'Location', 'Device', 'PlayTimeOfDay', 'TotalPlaytimeInHours', 'PurchaseHistory', 'InvolvementLevel', 'UserReview'], axis=1, inplace=True)
        user_df.to_pickle('data/prepared/user_df.pkl')
        logger.info("Users data processing complete.")

    def process_game_data(self):
        logger.info("Processing games data...")
        game_columns = ['GameTitle', 'GameGenre', 'GameUpdateFrequency', 'LoadingTimeInSeconds', 'GameSettingsPreference']
        game_df = self.df[game_columns].drop_duplicates(subset='GameTitle').reset_index(drop=True)
        game_df['CombinedColumn'] = game_df.apply(lambda row: ' '.join(map(str, row)), axis=1)
        game_df['CombinedColumn'] = game_df['CombinedColumn'].str.lower()
        game_df.drop(['GameGenre', 'GameUpdateFrequency', 'LoadingTimeInSeconds', 'GameSettingsPreference'], axis=1, inplace=True)
        game_df.to_pickle('data/prepared/game_df.pkl')

        game_description_columns = ['GameTitle', 'GameGenre']
        game_description_df = self.df[game_description_columns].drop_duplicates(subset='GameTitle').reset_index(drop=True)
        game_description_df['CombinedColumn'] = game_description_df.apply(lambda row: ' '.join(map(str, row)), axis=1)
        game_description_df['CombinedColumn'] = game_description_df['CombinedColumn'].str.lower()
        game_description_df.drop(['GameGenre'], axis=1, inplace=True)
        game_description_df.to_pickle('data/prepared/game_description_df.pkl')
        logger.info("Games data processing complete.")

    def process_user_history_data(self):
        logger.info("Processing users history data...")
        user_history_columns = ['UserID', 'GameTitle', 'Rating']
        user_history_df = self.df[user_history_columns].reset_index(drop=True)
        user_history_df.to_pickle('data/prepared/user_history_df.pkl')
        logger.info("Users history data processing complete.")

    def process_mean_ratings(self):
        logger.info("Processing mean rating...")
        mean_ratings = self.df.groupby('GameTitle')['Rating'].mean().reset_index()
        mean_ratings.to_pickle('data/prepared/mean_ratings.pkl')
        logger.info("Mean rating processing complete.")

    def process_all_data(self):
        self.process_user_data()
        self.process_game_data()
        self.process_user_history_data()
        self.process_mean_ratings()
        self.df.to_pickle('data/prepared/df.pkl')


@click.command()
@click.option('--file_path',
              type=click.Path(exists=True),
              default='data/raw/ml_test_rec_sys.parquet'
              )
def process_data(file_path):
    """Process data from a given file."""
    processor = DataProcessor(file_path)
    processor.process_all_data()


if __name__ == '__main__':

    process_data()
