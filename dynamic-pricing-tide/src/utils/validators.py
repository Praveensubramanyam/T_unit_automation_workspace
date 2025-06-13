def validate_input_data(data):
    """
    Validate the input data for the model.

    Parameters:
    - data: The input data to validate.

    Returns:
    - bool: True if the data is valid, False otherwise.
    """
    # Check if data is a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        print("Input data is not a pandas DataFrame.")
        return False

    # Check for missing values
    if data.isnull().values.any():
        print("Input data contains missing values.")
        return False

    # Add more validation checks as needed

    return True


def validate_model_parameters(params):
    """
    Validate the model parameters.

    Parameters:
    - params: The parameters to validate.

    Returns:
    - bool: True if the parameters are valid, False otherwise.
    """
    # Check if parameters are within expected ranges
    if params['learning_rate'] <= 0 or params['learning_rate'] > 1:
        print("Learning rate must be between 0 and 1.")
        return False

    if params['n_estimators'] <= 0:
        print("Number of estimators must be a positive integer.")
        return False

    # Add more validation checks as needed

    return True