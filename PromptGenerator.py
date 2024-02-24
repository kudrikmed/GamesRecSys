from deep_translator import GoogleTranslator
import logging

logging.basicConfig(
    filename='logs/recsys.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PromptGenerator:
    def __init__(self, recommendations_dict, user_input):
        self.recommendations_dict = recommendations_dict
        self.user_input = user_input
        self.logger = logging.getLogger(__name__)

    def translate_user_input(self):
        return GoogleTranslator(source='auto', target='en').translate(self.user_input)

    def prepare_prompt(self):
        translated_customer_input = self.translate_user_input()

        my_history = self.recommendations_dict['my_history']
        similar_games = self.recommendations_dict['similar_games']
        similar_users_history = self.recommendations_dict['similar_users_history']

        message_objects = []
        message_objects.append({"role": "system", "content": "You're a chatbot helping users to play interesting games"})
        message_objects.append({"role": "user", "content": translated_customer_input})
        message_objects.append({"role": "user", "content": f"Here're my latest top-rated games: {', '.join(my_history)}"})
        message_objects.append({"role": "user", "content": f"Please give me a detailed explanation of your recommendations"})
        message_objects.append({"role": "user", "content": "Please be friendly and talk to me like a person, don't just give me a list of recommendations"})
        message_objects.append({"role": "assistant", "content": f"I found these games I would recommend because they are similar to games you like: {', '.join(similar_games)}"})
        message_objects.append({"role": "assistant", "content": f"I found these games I would recommend because they are played by users with similar interests to you: {', '.join(similar_users_history)}"})
        message_objects.append({"role": "assistant", "content": "Here's my summarized recommendation of games, and why you will find them interesting:"})

        self.logger.info("Prompt prepared successfully.")
        return message_objects
