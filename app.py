import pickle
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and vectorizer
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None

@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        print(f"Error loading home page: {e}")
        return "An error occurred while loading the home page", 500

@app.route("/predict", methods=["POST"])
def predict():
    # For web form submissions only
    try:
        if model is None or vectorizer is None:
            return "Model or vectorizer not loaded properly", 500

        message = request.form.get("message")
        if not message:
            return "No message provided", 400

        input_data = vectorizer.transform([message])
        prediction = model.predict(input_data)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return render_template("index.html", prediction_text=f"Message is: {result}")

    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Internal Server Error", 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    # For API (Flutter/mobile/web) requests
    try:
        if model is None or vectorizer is None:
            return jsonify({"error": "Model or vectorizer not loaded properly"}), 500

        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400

        message = data["message"]
        input_data = vectorizer.transform([message])
        prediction = model.predict(input_data)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return jsonify({"result": result})

    except Exception as e:
        print(f"Error in API prediction: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
