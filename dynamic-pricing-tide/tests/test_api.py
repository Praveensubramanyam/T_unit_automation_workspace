import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_get_pricing():
    response = client.get("/pricing")
    assert response.status_code == 200
    assert "pricing" in response.json()

def test_post_pricing():
    response = client.post("/pricing", json={"item": "item_name", "quantity": 2})
    assert response.status_code == 201
    assert "price" in response.json()