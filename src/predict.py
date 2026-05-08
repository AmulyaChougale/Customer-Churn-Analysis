import joblib
import pandas as pd
from preprocessing import preprocess

data = joblib.load("churn_pipeline.pkl")

model = data["model"]
columns = data["columns"]

def predict_churn(input_data):
    
    # Convert multiple inputs into DataFrame
    df = pd.DataFrame(input_data)
    
    # Apply preprocessing
    df = preprocess(df)
    
    # Align columns
    df = df.reindex(columns=columns, fill_value=0)
    
    # Predict
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]


    prediction_labels = [
        "Churn" if pred == 1 else "Stay"
        for pred in predictions
    ]
    
    # Store results
    results = pd.DataFrame({
        "Prediction": prediction_labels,
        "Probability": probabilities
    })
    
    return results


if __name__ == "__main__":

    input_data = [

        {
            'CreditScore': 800,
            'Age': 35,
            'Tenure': 4,
            'Balance': 100000,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 70000,
            'Geography': 'Spain',
            'Gender': 'Female'
        },

        {
            'CreditScore': 420,
            'Age': 58,
            'Tenure': 2,
            'Balance': 150000,
            'NumOfProducts': 1,
            'HasCrCard': 1,
            'IsActiveMember': 0,
            'EstimatedSalary': 40000,
            'Geography': 'Germany',
            'Gender': 'Female'
        },

        {
            'CreditScore': 650,
            'Age': 40,
            'Tenure': 5,
            'Balance': 50000,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 0,
            'EstimatedSalary': 60000,
            'Geography': 'Germany',
            'Gender': 'Male'
        }
    ]

    results = predict_churn(input_data)

    print(results)