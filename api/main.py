import json
import os
import ssl
import urllib
from datetime import datetime
from typing import List

import numpy as np
from azure.data.tables import TableServiceClient
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

context = ssl._create_unverified_context()

# Load env vars
load_dotenv()

TABLE_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
TABLE_NAME = os.getenv("TABLE_NAME", "businessInsights")

service = TableServiceClient.from_connection_string(TABLE_CONNECTION_STRING)
table_client = service.get_table_client(table_name=TABLE_NAME)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AccuracyRecord(BaseModel):
    date: str
    value: float


class AccuracyResponse(BaseModel):
    accuracy: List[AccuracyRecord]


@app.get("/metrics/accuracy", response_model=AccuracyResponse)
def get_accuracy_metrics():
    try:
        entities = table_client.list_entities()

        records = []
        for entity in entities:
            try:
                raw_date = entity.get("timestamp") or entity.get("date")
                parsed_dt = (
                    raw_date
                    if isinstance(raw_date, datetime)
                    else datetime.fromisoformat(raw_date)
                )
                records.append(
                    {
                        "date": parsed_dt.isoformat(),
                        "value": float(entity["directional_accuracy"]),
                    }
                )
            except Exception:
                continue  # Skip malformed records

        # Sort and return last 10
        records.sort(key=lambda r: r["date"], reverse=True)
        latest_records = records[-10:]

        return {"accuracy": latest_records}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch accuracy metrics: {str(e)}",
        )


class PredictionInput(BaseModel):
    MRP: float
    DiscountRate: float
    Brand_encoded: int
    Is_Premium_Brand: int
    Is_Peak_Season: int
    Product_Age_Days: int
    IsMetroMarket: int
    UnitsSold: int
    CTR: float
    BounceRate: float
    ReturningVisitorRatio: float
    FunnelDrop_ViewToCart: float
    FunnelDrop_CartToCheckout: float
    AvgSessionDuration: int


def compute_features(data: PredictionInput):
    now = datetime.now()

    base_price = data.MRP * (1 - data.DiscountRate / 100)
    Log_MRP = float(np.log(base_price)) if base_price > 0 else 0.0

    MRP_Category = 2 if data.MRP > 1000 else (1 if data.MRP > 500 else 0)
    MRP_vs_Brand_Median = (
        1.0  # Placeholder, compute if brand median data is available
    )

    Is_Summer = 1 if now.month in [4, 5, 6] else 0
    IsWeekend = 1 if now.weekday() >= 5 else 0
    Is_Mid_Week = 1 if now.weekday() in [2, 3] else 0

    Month = now.month
    DayOfWeek = now.weekday()

    Is_New_Product = 1 if data.Product_Age_Days < 30 else 0
    Is_Mature_Product = 1 if data.Product_Age_Days > 180 else 0

    Lead_Time_Days = 3.5  # Placeholder
    Initial_Stock_Level = 1000  # Placeholder
    Historical_Demand_Mean_Brand_FC_ID = 250  # Placeholder
    Historical_Stock_Mean_Brand_FC_ID = 300  # Placeholder

    columns = [
        "Log_MRP",
        "MRP_Category",
        "MRP_vs_Brand_Median",
        "Is_Premium_Brand",
        "Product_Age_Days",
        "IsMetroMarket",
        "Brand_encoded",
        "Is_Peak_Season",
        "Is_Summer",
        "IsWeekend",
        "Is_Mid_Week",
        "Month",
        "DayOfWeek",
        "Is_New_Product",
        "Is_Mature_Product",
        "Lead_Time_Days",
        "Initial_Stock_Level",
        "Historical_Demand_Mean_Brand_FC_ID",
        "Historical_Stock_Mean_Brand_FC_ID",
    ]

    values = [
        Log_MRP,
        MRP_Category,
        MRP_vs_Brand_Median,
        data.Is_Premium_Brand,
        data.Product_Age_Days,
        data.IsMetroMarket,
        data.Brand_encoded,
        data.Is_Peak_Season,
        Is_Summer,
        IsWeekend,
        Is_Mid_Week,
        Month,
        DayOfWeek,
        Is_New_Product,
        Is_Mature_Product,
        Lead_Time_Days,
        Initial_Stock_Level,
        Historical_Demand_Mean_Brand_FC_ID,
        Historical_Stock_Mean_Brand_FC_ID,
    ]

    return {"input_data": {"columns": columns, "index": [0], "data": [values]}}


@app.post("/predict")
def predict_price(input_data: PredictionInput):
    try:
        model_input = compute_features(input_data)

        url = "https://t-unit-workspace-wvobj.centralindia.inference.ml.azure.com/score"

        body = str.encode(json.dumps(model_input))

        api_key = "4TcySwrfYZuDUlyRnWTU6jKqSdHmmWmMooPFsmJwe7it7xFJlt4gJQQJ99BFAAAAAAAAAAAAINFRAZML1xT9"
        if not api_key:
            raise Exception("A key should be provided to invoke the endpoint")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": ("Bearer " + api_key),
        }

        req = urllib.request.Request(url, body, headers)

        try:
            response = urllib.request.urlopen(req, context=context)

            result = response.read()
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))

        response_data = json.loads(result)
        price = response_data[0]

        return {
            "price": price,
            "revenue": price * input_data.UnitsSold,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        )
