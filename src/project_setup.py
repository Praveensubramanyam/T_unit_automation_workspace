import sys
from pathlib import Path

from utils.setup_utils import (
    setup_custom_exceptions,
    setup_key_vault_integration,
    setup_logging,
    setup_mlflow_tracking,
    setup_project_structure,
)

# Add 'src' directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent / "src"))


def main():
    print("Initializing Tide Dynamic Pricing project setup...\n")
    setup_project_structure()
    setup_custom_exceptions()
    secret_client = setup_key_vault_integration()
    azure_instrumentation_key = secret_client.get_secret(
        "Instrumentation-Key"
    ).value
    print(
        "Azure Application Insights Instrumentation Key retrieved successfully."
    )
    print(azure_instrumentation_key)
    setup_logging(azure_instrumentation_key=azure_instrumentation_key)
    setup_mlflow_tracking()

    print("\n✅ Project setup complete.")


if __name__ == "__main__":
    main()
