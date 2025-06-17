# dynamic-pricing-tide

This project implements a dynamic pricing model using machine learning techniques. The structure of the project is organized into several directories, each serving a specific purpose.

## Project Structure

- **data/**: Contains raw and processed datasets.
  - **raw/**: Stores raw datasets.
  - **processed/**: Stores processed datasets.

- **notebooks/**: Contains Jupyter Notebooks for exploration.
  - **01_data_exploration.ipynb**: Used for exploratory data analysis.

- **src/**: Contains the core source code.
  - **config/**: Contains configurations and constants.
    - **settings.py**: Configuration settings for the project.
  - **data/**: Handles data loading and preprocessing.
    - **preprocess.py**: Functions for loading and preprocessing datasets.
  - **features/**: Responsible for feature engineering.
    - **feature_builder.py**: Functions for creating features from the dataset.
  - **models/**: For model training and evaluation.
    - **train.py**: Functions for training machine learning models.
    - **evaluate.py**: Functions for evaluating the performance of trained models.
  - **monitor/**: For drift detection and logging.
    - **monitor.py**: Functions for monitoring model performance and detecting data drift.
  - **utils/**: Contains utility functions.
    - **logger.py**: Functions for logging information and errors.
    - **validators.py**: Validation functions for input data.

- **api/**: Contains the backend API built with FastAPI.
  - **main.py**: Entry point for the FastAPI application.
  - **routers/**: Contains API route definitions.
    - **pricing.py**: Defines routes related to pricing.

- **frontend/**: Contains the frontend application.
  - **streamlit_app.py**: Main entry point for the Streamlit application.

- **.github/**: Contains GitHub-related files.
  - **workflows/**: Contains CI/CD workflow configurations.
    - **ci_cd_pipeline.yml**: Defines the GitHub Actions CI/CD pipeline.

- **tests/**: Contains unit and integration tests.
  - **test_preprocessing.py**: Tests for the preprocessing functions.
  - **test_api.py**: Tests for the API endpoints.

- **requirements.txt**: Lists the Python dependencies required for the project.

- **.env**: Used for local environment variables.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/dynamic-pricing-tide.git
   cd dynamic-pricing-tide
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables in the `.env` file.

## Usage

- To run the API, navigate to the `api` directory and execute:
  ```
  uvicorn main:app --reload
  ```

- To run the frontend application, execute:
  ```
  streamlit run streamlit_app.py
  ```

- For data exploration, open the Jupyter Notebook in the `notebooks` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.