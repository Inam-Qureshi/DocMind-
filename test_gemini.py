
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# Load the project's .env file
load_dotenv(
    Path(__file__).resolve().parent / ".env",
    override=True
)

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL = os.getenv("GEMINI_MODEL", "").strip()

if not API_KEY or not MODEL:
    raise SystemExit("API key or model name is missing.")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

HEADERS = {
    "x-goog-api-key": API_KEY,
    "Content-Type": "application/json"
}

# Check whether the selected model supports text generation
print("Checking model:", MODEL)

models_response = requests.get(
    f"{BASE_URL}/models",
    headers=HEADERS,
    timeout=(10, 30)
)

print("Model-list status:", models_response.status_code)

if not models_response.ok:
    print(models_response.text[:1500])
    raise SystemExit("Could not retrieve available models.")

models = models_response.json().get("models", [])

selected_model = next(
    (
        item for item in models
        if item["name"] == f"models/{MODEL}"
    ),
    None
)

if selected_model is None:
    print("Model not found in the available model list.")
    print("Available generation models:")

    for item in models:
        if "generateContent" in item.get(
            "supportedGenerationMethods", []
        ):
            print(item["name"])

    raise SystemExit("Check your GEMINI_MODEL setting.")

if "generateContent" not in selected_model.get(
    "supportedGenerationMethods", []
):
    raise SystemExit("Selected model does not support generateContent.")

print("Model exists and supports text generation.")

# Test actual text generation
payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": "Reply only: Gemini connection works."
                }
            ]
        }
    ],
    "generationConfig": {
        "temperature": 0,
        "maxOutputTokens": 30
    }
}

try:
    response = requests.post(
        f"{BASE_URL}/models/{MODEL}:generateContent",
        headers=HEADERS,
        json=payload,
        timeout=(15, 120)
    )

    print("Generation status:", response.status_code)
    print("Generation response:", response.text[:1500])

except requests.exceptions.Timeout:
    print("Generation request timed out.")

except requests.exceptions.RequestException as error:
    print("Request failed:", error)
