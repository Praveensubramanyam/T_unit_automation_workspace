import os
import logging
import time
import dotenv
from json_log_formatter import JSONFormatter
from functools import wraps
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
    
def setup_logging(azure_instrumentation_key=None):
    logger = logging.getLogger("dynamic_pricing_tide")
    logger.setLevel(logging.DEBUG)

    formatter = JSONFormatter()
    
    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if azure_instrumentation_key:
        azure_handler = AzureLogHandler(
            connection_string=f'InstrumentationKey={azure_instrumentation_key}'
        )
    azure_handler.setFormatter(formatter)
    logger.addHandler(azure_handler)

    logger.info("✅ Logging setup completed.")
    print("✅ Logging setup completed successfully.")

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
    
def retry_on_exception(max_retries=3, delay=2, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f"Retrying {func.__name__} due to error: {e}")
                    time.sleep(delay)
            raise Exception(f"Max retries exceeded for {func.__name__}")
        return wrapper
    return decorator

def setup_key_vault_integration():
    key_vault_url = os.getenv("AZURE_KEY_VAULT_URL")
    if not key_vault_url:
        raise ValueError("AZURE_KEY_VAULT_URL not set.")
    
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=key_vault_url, credential=credential)

    try:
        secret_name = os.getenv("AZURE_TEST_SECRET")  # optional demo fetch
        if secret_name:
            secret = secret_client.get_secret(secret_name)
            print(f"Retrieved secret '{secret_name}' {secret} from Key Vault.")
    except Exception as e:
        print(f"Warning: Could not fetch test secret. {e}")
    
    print("✅ Key Vault integration setup completed.")
    return secret_client

class RateLimiter:
    def __init__(self, rate_per_second):
        self.interval = 1.0 / rate_per_second
        self.last_time = time.time()

    def wait(self):
        now = time.time()
        elapsed = now - self.last_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_time = time.time()

def setup_mlflow_tracking():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Tide Dynamic Pricing")

    print("MLflow tracking setup completed successfully.")
