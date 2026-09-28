import requests
import time

url = "http://localhost:8000/api/trigger/"
data = {
    "shop_name": "LRN Store",
    "camera_name": "LRN Store - Channel2",
    "model_name": "theft"
}

r1 = requests.post(url, data=data)
print("1st:", r1.json())
time.sleep(2)
r2 = requests.post(url, data=data)
print("2nd:", r2.json())
