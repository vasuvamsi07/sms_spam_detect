import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

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
    try:
        if model is None or vectorizer is None:
            return "Model or vectorizer not loaded properly", 500

        message = request.form.get("message")
        if not message:
            return "No message provided", 400  # If message is missing

        # Transform the message using the vectorizer
        input_data = vectorizer.transform([message])
        prediction = model.predict(input_data)[0]

        # Display prediction result
        result = "Spam" if prediction == 1 else "Not Spam"
        return render_template("index.html", prediction_text=f"Message is: {result}")
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Internal Server Error", 500  # Return HTTP 500 error with message

if __name__ == "__main__":
    app.run(debug=True)  # Enable debug mode for detailed error logs
