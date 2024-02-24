FROM python:3.9

COPY requirements.txt /workdir/
COPY app.py GetRecommendations.py PromptGenerator.py LLMRecommendator.py CheckUserInput.py /workdir/
COPY models/ /workdir/models/
COPY data/prepared/ /workdir/data/prepared
COPY logs/get_recommendations.log /workdir/logs/

WORKDIR /workdir

RUN pip install -r requirements.txt

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]