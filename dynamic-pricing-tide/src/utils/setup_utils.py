import os
import logging
import dotenv
from json_log_formatter import JSONFormatter
from opencensus.ext.azure.log_exporter import AzureLogHandler
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import mlflow

dotenv.load_dotenv()

def setup_project_structure():
    folders = [
        "data/raw/",
        "data/processed/",
        "notebooks",
        "src/config",
        "src/data",
        "src/features",
        "src/models",
        "src/monitor",
        "src/utils",
        "api/routers",
        "frontend",
        ".github/workflows",
        "tests"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created directory: {folder}")

    print("Project structure initialized successfully.")
    
def setup_logging():
    logger = logging.getLogger("dynamic_pricing_tide")
    logger.setLevel(logging.DEBUG)

    formatter = JSONFormatter()
    
    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # azure_handler = AzureLogHandler(
    #     workspace_id="YOUR_WORKSPACE_ID",
    #     shared_key="YOUR_SHARED_KEY",
    #     log_type="DynamicPricingTideLogs"
    # )
    # azure_handler.setFormatter(formatter)
    # logger.addHandler(azure_handler)

    print("Logging setup completed successfully.")

def setup_custom_exceptions():
    class DataValidationError(Exception):
        """Exception raised for errors in the input data validation."""
        def __init__(self, message):
            self.message = message
            super().__init__(self.message)

    class ModelTrainingException(Exception):
        """Exception raised for errors during model training."""
        def __init__(self, message):
            self.message = message
            super().__init__(self.message)

    class PredictionException(Exception):
        """Exception raised for errors during prediction."""
        def __init__(self, message):
            self.message = message
            super().__init__(self.message)

    print("Custom exceptions setup completed successfully.")

def setup_key_vault_integration():
    key_vault_url = os.getenv("AZURE_KEY_VAULT_URL")
    if not key_vault_url:
        raise ValueError("AZURE_KEY_VAULT_URL environment variable is not set.")
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=key_vault_url, credential=credential)

    print("Key Vault integration setup completed successfully.")
    return secret_client

def setup_mlflow_tracking():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Tide Dynamic Pricing")

    print("MLflow tracking setup completed successfully.")
