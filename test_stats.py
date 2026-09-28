import sys
from fastapi.testclient import TestClient

sys.path.append("/home/ai-mini-playback/AI-Server/backend")
from app.main import app

client = TestClient(app)
response = client.get("/api/dashboard/stats")
print(response.status_code)
print(response.json())
