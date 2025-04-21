import re
import joblib
import nltk
import string
import pickle
import cv2
import numpy as np
import base64
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
from io import BytesIO

# Creating flask app
app = Flask(__name__)

# Enable CORS for the Flask app
CORS(app)  # Applies CORS to all routes

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
post_detection = joblib.load("./model/post_detection.joblib")

# Download necessary materials
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words("english"))

# Reading the model files
with open('./model/cv_final.pkl', 'rb') as file:
    cv = pickle.load(file)
with open('./model/model_final.pkl', 'rb') as file:
    model = pickle.load(file)


# Defining the text preprocessing function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


def preprocess_image(img):
    roi = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224, 224))
    roi = preprocess_input(roi)
    roi = np.expand_dims(roi, axis=0)
    features = base_model.predict(roi)
    features = features.flatten()
    output = post_detection.predict([features])
    return output


@app.route('/detect_comment', methods=["POST"])
def getData():
    try:
        # Get the JSON data from the request
        data = request.json
        if data is None:
            raise ValueError("No JSON data received")

        # Extract the message from the JSON data
        message = data.get("message")
        if not message:
            return jsonify({"Error": "The 'message' key is missing or empty in the request!"})

        # Clean and transform the message
        cleaned_message = clean_text(message)
        df = cv.transform([cleaned_message]).toarray()

        # Make a prediction
        prediction = model.predict(df)[0]

        # Return the result as a JSON response
        return jsonify({"result": prediction})

    except Exception as e:
        # Handle any exceptions and return an error message
        return jsonify({"Error": str(e)})


# Endpoint to detect inappropriate content in images
@app.route('/detect_post', methods=["POST"])
def detect_post():
    try:
        # Get the JSON data from the request
        data = request.json
        if data is None:
            raise ValueError("No JSON data received")

        # Extract the base64-encoded image from the JSON data
        img_data = data.get("image")
        if not img_data:
            return jsonify({"Error": "The 'image' key is missing or empty in the request!"})

        # Decode the base64-encoded image
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))
        img_np = np.array(img)

        # Preprocess the image and get the prediction
        prediction = preprocess_image(img_np)[0]
        prediction = "non_violence" if prediction == 0 else "violence"

        # Return the prediction result
        return jsonify({"result": prediction})

    except Exception as e:
        # Handle any exceptions and return an error message
        return jsonify({"Error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
