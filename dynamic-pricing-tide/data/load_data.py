# src/data/load_data.py

from pyspark.sql import SparkSession

def load_delta_table(table_name: str):
    """
    Load a Delta table from Unity Catalog in Azure Databricks.
    
    Args:
        table_name (str): Full Unity Catalog table path (e.g., "dbx-azure-catalog.t_unit.sales_features_combined")
    
    Returns:
        pyspark.sql.DataFrame: Spark DataFrame of the Delta table
    """
    spark = SparkSession.builder.appName("LoadDeltaTable").getOrCreate()
    
    try:
        df = spark.read.format("delta").table(table_name)
        print(f"✅ Successfully loaded table: {table_name}")
        return df
    except Exception as e:
        print(f"❌ Failed to load table {table_name}: {e}")
        raise
