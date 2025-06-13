import os

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

def create_folders():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created directory: {folder}")

if __name__ == "__main__":
    create_folders()
    print("Folder structure initialized successfully.")