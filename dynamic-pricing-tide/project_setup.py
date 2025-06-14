import sys
from pathlib import Path

# Add 'src' directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from utils.setup_utils import (
    setup_project_structure,
    setup_logging,
    setup_custom_exceptions,
    setup_key_vault_integration,
    setup_mlflow_tracking
)
def main():
    print("Initializing Tide Dynamic Pricing project setup...\n")
    setup_project_structure()
    setup_logging()
    setup_custom_exceptions()
    secret_client = setup_key_vault_integration()
    setup_mlflow_tracking()

    print("\n✅ Project setup complete.")
    
if __name__ == "__main__":
    main()