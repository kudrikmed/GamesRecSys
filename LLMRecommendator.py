import requests


class TextRecommendationAPI:
    def __init__(self, url=r"http://localhost:1337/v1/chat/completions"):
        self.url = url

    def get_text_recommendations(self, message, model="gpt-4-0613", stream=False):
        body = {
            "model": model,
            "stream": stream,
            "messages": message
        }
        try:
            json_response = requests.post(self.url, json=body).json().get('choices', [])
            for choice in json_response:
                return choice.get('message', {}).get('content', '')
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            return None
