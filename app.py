from fastapi import FastAPI
import logging
from GetRecommendations import GameRecommendationSystem
from PromptGenerator import PromptGenerator
from LLMRecommendator import TextRecommendationAPI
from CheckUserInput import predict_question

app = FastAPI()

logging.basicConfig(
    filename='logs/recsys.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@app.get("/recommend/")
async def recommend_games(user_id: str, user_input: str):
    """
    Endpoint to recommend games based on user input.

    Parameters:
    - user_id (str): The unique identifier of the user.
    - user_input (str): The input provided by the user.

    Returns:
    - dict: A dictionary containing recommendations and, if applicable, text recommendations.
    """

    logger.info(f"Received recommendation request for user_id: {user_id}, user_input: {user_input}")

    game_recommender = GameRecommendationSystem()
    recommendations = game_recommender.make_recommendations(user_id)

    if predict_question(user_input):
        recommendation_system = PromptGenerator(recommendations, user_input)
        message = recommendation_system.prepare_prompt()

        api = TextRecommendationAPI()
        result = api.get_text_recommendations(message)

        if result is not None:
            logger.info("Text recommendation retrieved successfully.")
            return {"text_recommendation": result,
                    "recommendations": recommendations}
        else:
            logger.warning("Failed to retrieve text recommendation.")
            return {"text_recommendation": "Failed to retrieve text recommendation.",
                    "recommendations": recommendations}
    else:
        logger.info("Non-game-related question detected.")
        return {"text_recommendation": "It looks like your question is not about games. Try rephrasing it.",
                "recommendations": recommendations}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
