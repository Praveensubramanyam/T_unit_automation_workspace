import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

def load_data(data_path):
    return pd.read_csv(data_path)

def train_model(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    joblib.dump(model, model_path)

def main(data_path, model_output_path):
    # Load dataset
    data = load_data(data_path)
    
    # Assume the last column is the target variable
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    print(f'Mean Squared Error: {mse}')
    
    # Save the model
    save_model(model, model_output_path)

if __name__ == "__main__":
    data_path = os.path.join('data', 'processed', 'your_dataset.csv')  # Update with your dataset path
    model_output_path = os.path.join('models', 'trained_model.pkl')
    main(data_path, model_output_path)