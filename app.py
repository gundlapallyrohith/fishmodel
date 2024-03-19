from flask import Flask, request, render_template
import joblib

# Create the Flask app
app = Flask(__name__)

# Load the regression model (make sure to provide the correct path)
regressor = joblib.load('fish_weight_regressor.pkl')

# Define the home page route
@app.route('/')
def home():
    return render_template('input.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input features from the form, making sure none are missing
        input_features = []
        for feature_name in ['Length1', 'Length2', 'Length3', 'Height', 'Width']:
            feature_value = request.form.get(feature_name)
            if feature_value is None:
                raise ValueError(f"The feature '{feature_name}' is missing. Please provide all required inputs.")
            input_features.append(float(feature_value))
        
        # Make prediction with the model
        prediction = regressor.predict([input_features])[0]
        predicted_weight = round(prediction, 2)
        
        # Return the result page with the prediction text
        return render_template('result.html', prediction_text=f'The predicted fish weight is {predicted_weight} grams')
    except Exception as e:
        # Print the error and return the input form with the error message
        print(e)
        return render_template('input.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
