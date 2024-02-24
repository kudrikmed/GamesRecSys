import unittest
import os
import pandas as pd
import tempfile
import logging
from PreprocessData import DataProcessor


class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.sample_data = {
            'UserID': [1, 2, 3],
            'Age': [25, 30, 22],
            'Gender': ['Male', 'Female', 'Male'],
            'Location': ['City', 'Suburb', 'City'],
            'Device': ['PC', 'Console', 'PC'],
            'PlayTimeOfDay': ['Morning', 'Evening', 'Night'],
            'TotalPlaytimeInHours': [10, 20, 15],
            'PurchaseHistory': ['GameA', 'GameB', 'GameA'],
            'InvolvementLevel': ['High', 'Medium', 'Low'],
            'UserReview': [4.5, 3.0, 2.5],
            'GameTitle': ['GameA', 'GameB', 'GameC'],
            'GameGenre': ['Action', 'Adventure', 'Action'],
            'GameUpdateFrequency': ['Monthly', 'Weekly', 'Daily'],
            'LoadingTimeInSeconds': [30, 40, 20],
            'GameSettingsPreference': ['High', 'Medium', 'Low'],
            'Rating': [4.0, 3.5, 2.0]
        }

        self.df = pd.DataFrame(self.sample_data)

        # Create a temporary directory and save the Parquet file
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file_path = os.path.join(self.temp_dir, 'test.parquet')
        self.df.to_parquet(self.temp_file_path)

    def test_process_user_data(self):
        processor = DataProcessor(os.path.join(self.temp_dir, 'test.parquet'))
        processor.df = self.df
        processor.process_user_data()

        # Check if the processed user data file exists
        self.assertTrue(os.path.exists('data/prepared/user_df.pkl'))

    def test_process_game_data(self):
        processor = DataProcessor(os.path.join(self.temp_dir, 'test.parquet'))
        processor.df = self.df
        processor.process_game_data()

        # Check if the processed game data files exist
        self.assertTrue(os.path.exists('data/prepared/game_df.pkl'))
        self.assertTrue(os.path.exists('data/prepared/game_description_df.pkl'))

    def test_process_user_history_data(self):
        processor = DataProcessor(os.path.join(self.temp_dir, 'test.parquet'))
        processor.df = self.df
        processor.process_user_history_data()

        # Check if the processed user history data file exists
        self.assertTrue(os.path.exists('data/prepared/user_history_df.pkl'))

    def test_process_mean_ratings(self):
        processor = DataProcessor(os.path.join(self.temp_dir, 'test.parquet'))
        processor.df = self.df
        processor.process_mean_ratings()

        # Check if the processed mean ratings file exists
        self.assertTrue(os.path.exists('data/prepared/mean_ratings.pkl'))

    def tearDown(self):
        # Clean up created files after tests
        file_list = [
            'data/prepared/user_df.pkl',
            'data/prepared/game_df.pkl',
            'data/prepared/game_description_df.pkl',
            'data/prepared/user_history_df.pkl',
            'data/prepared/mean_ratings.pkl',
        ]
        for file_path in file_list:
            if os.path.exists(file_path):
                os.remove(file_path)

        if os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)


if __name__ == '__main__':
    unittest.main()
