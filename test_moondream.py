import base64
import requests
import json

with open("dog.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "moondream",
    "prompt": "What animal is in this image?",
    "images": [img_b64],
    "stream": False,
    "options": {
        "temperature": 0
    }
}

response = requests.post(
    "http://127.0.0.1:11434/api/generate",
    json=payload,
    timeout=120
)

print(json.dumps(response.json(), indent=2))