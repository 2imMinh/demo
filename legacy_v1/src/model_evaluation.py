import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def model_evaluation(y_true, y_pred):
    """
    Evaluate model performance using various metrics.
    
    Args:
        y_true: DataFrame or array of true values
        y_pred: DataFrame or array of predicted values
    """
    # Convert DataFrames to numpy arrays if needed
    if hasattr(y_true, 'values'):
        y_true = y_true.values.flatten()
    else:
        y_true = np.array(y_true).flatten()
    
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values.flatten()
    else:
        y_pred = np.array(y_pred).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Print evaluation results
    print("=" * 50)
    print("Model Evaluation Results")
    print("=" * 50)
    print(f"Mean Absolute Error (MAE):     {mae:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"R-squared (R²):                 {r2:.6f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("=" * 50)
