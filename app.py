from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open('models/xgb_classifier_model.pkl', 'rb') as model_file:
    xgb_classifier = pickle.load(model_file)

with open('models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Mapping of numerical predictions to poker hand rankings
poker_hand_ranking = {
    0: "Nothing in hand",
    1: "One pair",
    2: "Two pairs",
    3: "Three of a kind",
    4: "Straight",
    5: "Flush",
    6: "Full house",
    7: "Four of a kind",
    8: "Straight flush",
    9: "Royal flush"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [
        request.form['card1_rank'], request.form['card1_suit'],
        request.form['card2_rank'], request.form['card2_suit'],
        request.form['card3_rank'], request.form['card3_suit'],
        request.form['card4_rank'], request.form['card4_suit'],
        request.form['card5_rank'], request.form['card5_suit']
    ]
    input_data = [int(x) for x in input_data]
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = xgb_classifier.predict(input_data_scaled)
    predicted_hand = poker_hand_ranking[prediction[0]]
    return render_template('index.html', prediction_text=f'Predicted Poker Hand: {predicted_hand}')

if __name__ == '__main__':
    app.run(debug=True)
