from io import BytesIO, StringIO
from typing import Dict, List

import numpy as np
import pandas as pd
from azure.storage.blob import BlobServiceClient


class PricingDataValidator:
    """Validates pricing data for schema, missing values, and outliers."""

    def __init__(self, required_columns: List[str], price_column: str):
        self.required_columns = required_columns
        self.price_column = price_column

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate if all required columns are present."""
        missing_cols = [
            col for col in self.required_columns if col not in df.columns
        ]
        return {
            "missing_columns": missing_cols,
            "is_valid": len(missing_cols) == 0,
        }

    def validate_missing(self, df: pd.DataFrame) -> Dict[str, any]:
        """Check for missing values in the dataset."""
        missing = df.isnull().sum()
        return {
            "missing_counts": missing.to_dict(),
            "has_missing": missing.any(),
        }

    def detect_outliers(
        self, df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5
    ) -> pd.Series:
        """Detect outliers in price column using IQR or Z-score method."""
        if self.price_column not in df.columns:
            return pd.Series([False] * len(df))

        if method == "iqr":
            q1 = df[self.price_column].quantile(0.25)
            q3 = df[self.price_column].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            return ~df[self.price_column].between(lower, upper)
        elif method == "zscore":
            mean = df[self.price_column].mean()
            std = df[self.price_column].std()
            if std == 0:
                return pd.Series([False] * len(df))
            z_scores = (df[self.price_column] - mean) / std
            return np.abs(z_scores) > threshold
        else:
            raise ValueError(
                "Unsupported outlier detection method. Use 'iqr' or 'zscore'."
            )


class PreprocessingPipeline:
    """Handles data preprocessing including missing values and outlier removal."""

    def __init__(self, price_column: str):
        self.price_column = price_column

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate strategies for different column types."""
        df_copy = df.copy()

        for col in df_copy.columns:
            if df_copy[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
                elif pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    df_copy[col].fillna(method="ffill", inplace=True)
                else:
                    mode_val = df_copy[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else "missing"
                    df_copy[col].fillna(fill_val, inplace=True)

        return df_copy

    def remove_outliers(
        self, df: pd.DataFrame, outlier_mask: pd.Series
    ) -> pd.DataFrame:
        """Remove outliers based on the provided mask."""
        if outlier_mask.any():
            return df[~outlier_mask].copy()
        return df


class BlobStorageManager:
    """Manages Azure Blob Storage operations."""

    def __init__(self, connection_string: str):
        self.blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

    def download_blob_as_df(
        self, container: str, filename: str
    ) -> pd.DataFrame:
        """Download blob as pandas DataFrame."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container, blob=filename
            )
            stream = blob_client.download_blob().readall()
            return pd.read_csv(BytesIO(stream))
        except Exception as e:
            raise

    def upload_df_to_blob(
        self, df: pd.DataFrame, container: str, blob_name: str
    ):
        """Upload DataFrame to blob storage as CSV."""
        try:
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            blob_client = self.blob_service_client.get_blob_client(
                container=container, blob=blob_name
            )
            blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True)
        except Exception as e:
            raise


def get_price_columns(df: pd.DataFrame) -> List[str]:
    """Identify potential price columns in the dataset."""
    price_keywords = [
        "price",
        "cost",
        "amount",
        "selling",
        "revenue",
        "value",
        "fee",
    ]
    price_cols = []

    for col in df.columns:
        if any(keyword in col.lower() for keyword in price_keywords):
            if pd.api.types.is_numeric_dtype(df[col]):
                price_cols.append(col)

    return price_cols


class DataPreprocessor:
    """Main preprocessing class that orchestrates the entire preprocessing pipeline."""

    def __init__(self, blob_connection_string: str):
        self.blob_manager = BlobStorageManager(blob_connection_string)
        self.input_container = "dataset/raw"
        self.output_container = "dataset/processed"
        self.merged_container = "dataset/merged"

        self.file_list = [
            "Competitor_Pricing_Data.csv",
            "Daily_Customer_Behavior.csv",
            "Inventory_Data.csv",
            "Sales_Data.csv",
        ]

    def preprocess_individual_files(self):
        """Preprocess each file individually."""
        for filename in self.file_list:
            try:
                df = self.blob_manager.download_blob_as_df(
                    self.input_container, filename
                )
                price_cols = get_price_columns(df)

                if price_cols:
                    price_column = price_cols[0]
                    validator = PricingDataValidator(
                        required_columns=list(df.columns),
                        price_column=price_column,
                    )

                    schema_validation = validator.validate_schema(df)

                    missing_validation = validator.validate_missing(df)

                    outlier_mask = validator.detect_outliers(df)

                    pipeline = PreprocessingPipeline(price_column=price_column)
                    df = pipeline.handle_missing(df)
                    df = pipeline.remove_outliers(df, outlier_mask)
                else:
                    for col in df.columns:
                        if df[col].isnull().any():
                            if pd.api.types.is_numeric_dtype(df[col]):
                                df[col].fillna(df[col].median(), inplace=True)
                            else:
                                df[col].fillna("missing", inplace=True)

                cleaned_name = filename.replace(".csv", "_cleaned.csv")
                self.blob_manager.upload_df_to_blob(
                    df, self.output_container, cleaned_name
                )
            except Exception as e:
                continue

    def merge_datasets(self) -> pd.DataFrame:
        """Merge all cleaned datasets into a unified dataset."""
        try:
            sales_df = self.blob_manager.download_blob_as_df(
                self.output_container, "Sales_Data_cleaned.csv"
            )
            competitor_df = self.blob_manager.download_blob_as_df(
                self.output_container, "Competitor_Pricing_Data_cleaned.csv"
            )
            customer_df = self.blob_manager.download_blob_as_df(
                self.output_container, "Daily_Customer_Behavior_cleaned.csv"
            )
            inventory_df = self.blob_manager.download_blob_as_df(
                self.output_container, "Inventory_Data_cleaned.csv"
            )

            date_column_mapping = [
                (sales_df, "TransactionDate"),
                (competitor_df, "Date"),
                (customer_df, "Date"),
                (inventory_df, "Date"),
            ]

            for df, col in date_column_mapping:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            merged_df = sales_df.rename(columns={"TransactionDate": "Date"})
            merged_df = merged_df.merge(competitor_df, on="Date", how="left")
            merged_df = merged_df.merge(customer_df, on="Date", how="left")
            merged_df = merged_df.merge(inventory_df, on="Date", how="left")

            self.blob_manager.upload_df_to_blob(
                merged_df, self.merged_container, "unified_dataset.csv"
            )
            return merged_df
        except Exception as e:
            raise

    def run_complete_preprocessing(self) -> pd.DataFrame:
        """Run the complete preprocessing pipeline."""
        self.preprocess_individual_files()
        merged_df = self.merge_datasets()
        return merged_df
