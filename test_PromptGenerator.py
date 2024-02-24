import unittest
from unittest.mock import patch, MagicMock
from PromptGenerator import PromptGenerator


class TestPromptGenerator(unittest.TestCase):
    def setUp(self):
        self.recommendations_dict = {
            'my_history': ['Game1', 'Game2'],
            'similar_games': ['Game3', 'Game4'],
            'similar_users_history': ['Game5', 'Game6']
        }
        self.user_input = "Translate this sentence."

    def test_translate_user_input(self):
        with patch('PromptGenerator.GoogleTranslator') as mock_translator:
            mock_translate_instance = mock_translator.return_value
            mock_translate_instance.translate.return_value = "Translated sentence."

            prompt_generator = PromptGenerator(self.recommendations_dict, self.user_input)
            result = prompt_generator.translate_user_input()

            mock_translate_instance.translate.assert_called_once_with(self.user_input)
            self.assertEqual(result, "Translated sentence.")

    def test_prepare_prompt(self):
        prompt_generator = PromptGenerator(self.recommendations_dict, self.user_input)
        prompt_generator.translate_user_input = MagicMock(return_value="Translated sentence.")
        result = prompt_generator.prepare_prompt()

        self.assertEqual(len(result), 8)
        self.assertEqual(result[0]['role'], "system")
        self.assertEqual(result[1]['role'], "user")
        prompt_generator.translate_user_input.assert_called_once()


if __name__ == '__main__':
    unittest.main()
