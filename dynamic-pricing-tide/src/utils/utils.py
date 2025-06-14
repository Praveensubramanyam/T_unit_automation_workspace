# utils.py
"""
Utility module for the Dynamic Pricing project.
Includes:
- Structured logging with JSON formatting and Azure Application Insights integration
- Error handling with custom exception classes and retry decorators
- Azure Key Vault integration for secure secrets management
- Reusable utility functions like rate limiters and data validators
"""

import logging
import json
import time
from typing import Callable, Any
from functools import wraps
from logging import Logger
from logging.handlers import RotatingFileHandler

try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient
    from opencensus.ext.azure.log_exporter import AzureLogHandler
except ImportError:
    DefaultAzureCredential = None
    SecretClient = None
    AzureLogHandler = None

# Structured Logger

def get_logger(name: str, log_file: str = None, azure_instrumentation_key: str = None) -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    log_format = json.dumps({
        "time": "%(asctime)s",
        "level": "%(levelname)s",
        "name": "%(name)s",
        "message": "%(message)s"
    })

    formatter = logging.Formatter(log_format)

    if log_file:
        handler = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=3)
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if azure_instrumentation_key and AzureLogHandler:
        azure_handler = AzureLogHandler(connection_string=f'InstrumentationKey={azure_instrumentation_key}')
        logger.addHandler(azure_handler)

    return logger

# Error Handling

class ProjectError(Exception):
    """Base exception for project errors."""
    pass

class RetryableError(ProjectError):
    """Exception for retryable errors."""
    pass

def retry_on_exception(retries: int = 3, delay: float = 1.0, exceptions=(RetryableError,)):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt < retries - 1:
                        time.sleep(delay)
                    else:
                        raise
        return wrapper
    return decorator

# Azure Key Vault Integration

def get_secret_from_keyvault(vault_url: str, secret_name: str) -> str:
    if not (DefaultAzureCredential and SecretClient):
        raise ImportError("Azure SDK packages not installed.")
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)
    return client.get_secret(secret_name).value

# Utility Functions

def rate_limiter(max_calls: int, period: float):
    calls = []
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [call for call in calls if call > now - period]
            if len(calls) >= max_calls:
                raise RetryableError("Rate limit exceeded")
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_data(data: Any, schema: dict) -> bool:
    """Simple data validator against a schema (dict of key:type)."""
    if not isinstance(data, dict):
        return False
    for key, typ in schema.items():
        if key not in data or not isinstance(data[key], typ):
            return False
    return True
