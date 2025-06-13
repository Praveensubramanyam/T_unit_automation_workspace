# Contents of /ml-project/ml-project/src/main.py

import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config_path = os.path.join(os.path.dirname(__file__), '../configs/config.yaml')
    config = load_config(config_path)
    
    # Load data, train model, and make predictions here
    print("Loaded configuration:", config)

if __name__ == "__main__":
    main()