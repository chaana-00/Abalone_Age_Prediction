from operator import le
from flask import Flask, render_template, request
import pandas as pd
import joblib
from catboost import Pool
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('best_model_V13')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Example input data from the form
        input_data = {
            'Sex': request.form['Sex'],
            'Length': float(request.form['Length']),
            'Diameter': float(request.form['Diameter']),
            'Height': float(request.form['Height']),
            'Whole weight': float(request.form['Whole_weight']),
            'Whole weight.1': float(request.form['Whole_weight_1']),
            'Whole weight.2': float(request.form['Whole_weight_2']),
            'Shell weight': float(request.form['Shell_weight'])
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Initialize LabelEncoder
        le = LabelEncoder()

        le.fit(['Male', 'Female', 'Infant'])

        # Encode categorical variables
        input_df['Sex'] = le.transform(input_df['Sex'])

        # Convert DataFrame to Pool
        pool = Pool(data=input_df)
        
        # Prediction
        prediction = model.predict(pool)
        
        return render_template('result.html', prediction=prediction[0])
    
    except Exception as e:
        # Return the error message to help with debugging
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
