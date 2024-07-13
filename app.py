from flask import Flask, request, jsonify
from transformers import pipeline

# Load sentiment analysis pipeline from Hugging Face Transformers
sentiment_pipeline = pipeline("sentiment-analysis")

app = Flask(__name__)

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json(force=True)
    text = data['text']

    # Execute sentiment analysis
    results = sentiment_pipeline(text)

    # Prepare response
    response = {
        'text': text,
        'sentiment': results[0]['label'],
        'score': results[0]['score']
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)