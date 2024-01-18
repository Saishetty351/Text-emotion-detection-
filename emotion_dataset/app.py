import pickle
from pyexpat import model
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import altair as alt
import joblib

app = Flask(__name__)


# Load your trained model and emoji dictionary
model = pickle.load(open("text_emotion.pkl", "rb"))

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        raw_text = request.form["text"]
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)
        emoji_icon = emotions_emoji_dict[prediction]
        return render_template(
            "result.html",
            raw_text=raw_text,
            prediction=prediction,
            emoji_icon=emoji_icon,
            confidence=np.max(probability),
        )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
