import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

from process_data_pm25 import process_data as pm

def train_and_validate():
    time, positions, targets = pm()
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(positions, targets, test_size=0.1, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)

    y_val_pred = model.predict(X_val)
    mse_val = mean_squared_error(y_val, y_val_pred)
    rmse_val = np.sqrt(mse_val)

    print(f"Training MSE: {mse_train}, RMSE: {rmse_train}")
    print(f"Validation MSE: {mse_val}, RMSE: {rmse_val}")

    return model

# Save results

def save_results(X_train, y_train, y_train_pred, X_val, y_val, y_val_pred, model):
    with open('train_data.pkl', 'wb') as f:
        pickle.dump((X_train, y_train, y_train_pred), f)
    with open('validation_data.pkl', 'wb') as f:
        pickle.dump((X_val, y_val, y_val_pred), f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_and_validate()

