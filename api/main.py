from fastapi import FastAPI
from routers import pricing

app = FastAPI()

app.include_router(pricing.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Dynamic Pricing API!"}