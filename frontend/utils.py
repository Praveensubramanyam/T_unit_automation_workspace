from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

def get_secret_from_keyvault(vault_url: str, secret_name: str) -> str:
    if not (DefaultAzureCredential and SecretClient):
        raise ImportError("Azure SDK packages not installed.")
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)
    return client.get_secret(secret_name).value
