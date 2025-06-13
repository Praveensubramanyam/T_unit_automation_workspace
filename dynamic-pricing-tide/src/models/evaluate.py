def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of the trained model using various metrics.

    Parameters:
    - model: The trained model to evaluate.
    - X_test: The test features.
    - y_test: The true labels for the test set.

    Returns:
    - metrics: A dictionary containing evaluation metrics.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

    return metrics

def save_evaluation_results(metrics, filepath):
    """
    Save the evaluation metrics to a file.

    Parameters:
    - metrics: A dictionary containing evaluation metrics.
    - filepath: The path to the file where metrics will be saved.
    """
    import json

    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)