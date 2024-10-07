from flask import Flask, request, jsonify
import pickle

# Create Flask app
app = Flask(__name__)

# Load the pre-trained CountVectorizer and RandomForest model
with open('countvector.pkl', 'rb') as f:
    countvector = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    randomclassifier = pickle.load(f)

# Define a route for predicting stock sentiment


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data (headlines) from the POST request
        data = request.json['headlines']

        # Preprocess the input data using the same vectorizer used for training
        transformed_data = countvector.transform([data])

        # Predict using the pre-trained RandomForest classifier
        prediction = randomclassifier.predict(transformed_data)

        # Return the result as JSON
        return jsonify({
            'prediction': int(prediction[0])
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        })


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
