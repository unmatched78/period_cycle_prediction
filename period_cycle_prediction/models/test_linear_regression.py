import sys
import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# Add the parent directory to the Python path
sys.path.append('../')  # Adjust the path to locate the module correctly

from linear_regression import LinearRegression, utils  # Import after fixing the path

@pytest.fixture
def load_test_data():
    """Fixture to load the synthetic dataset."""
    df = pd.read_csv('../dataset/synthetic_data.csv', sep=',', header=0)
    return df

def test_data_preprocessing(load_test_data):
    """Test data preprocessing functions."""
    df = load_test_data
    periods_data = utils.calculate_datatime(df)
    features, labels = utils.generate_final_features(df)
    
    assert features is not None, "Features should not be None"
    assert labels is not None, "Labels should not be None"
    assert len(features) == len(labels), "Features and labels should have the same length"

def test_model_training_and_prediction(load_test_data):
    """Test the training and prediction of the Linear Regression model."""
    df = load_test_data
    periods_data = utils.calculate_datatime(df)
    features, labels = utils.generate_final_features(df)

    # Split the data
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=10)

    # Reshape the data
    train_x = np.array(x_train).reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    train_y = np.array(y_train).reshape((y_train.shape[0], y_train.shape[1] * 1))
    test_x = np.array(x_test).reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    test_y = np.array(y_test).reshape((y_test.shape[0], y_test.shape[1] * 1))

    # Train the model
    model_LR = LinearRegression()
    model_LR.fit(train_x, train_y)

    # Make predictions
    y_pred = model_LR.predict(test_x)

    # Check predictions
    assert len(y_pred) == len(test_y), "Predictions and test labels should have the same length"

    # Calculate RMSE
    rms = sqrt(mean_squared_error(test_y, y_pred))
    assert rms > 0, "RMSE should be greater than 0"

    # Calculate MAE
    mae = np.mean(np.abs((test_y - y_pred)))
    assert mae > 0, "MAE should be greater than 0"

    print("RMSE:", rms)
    print("MAE:", mae)