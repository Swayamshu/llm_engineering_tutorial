import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import httpx

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")
elif api_key.strip() != api_key:
    raise ValueError("API Key contains whitespace. Please remove it.")
else:
    print("API Key loaded successfully.\n")


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-3.6-flash"

clean_http_client = httpx.Client(trust_env=False)
gemini = OpenAI(
    base_url=GEMINI_BASE_URL,
    api_key=api_key,
    http_client=clean_http_client
)

message = "Hello Gemini! This is my first message."
messages=[{"role": "user", "content": message}]

response = gemini.chat.completions.create(
    model=GEMINI_MODEL,
    messages=messages
)

print(response.choices[0].message.content)

