# azure_test.py
import os
from pathlib import Path
from dotenv import load_dotenv
from azure_client import get_client

# Load .env from this folder
here = Path(__file__).resolve().parent
load_dotenv(dotenv_path=str(here / ".env"), override=True)

print("Loaded KEY =", os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY"))
print("Loaded ENDPOINT =", os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE"))

client = get_client()
print("Running test chat...")

resp = client.chat.completions.create(
    model=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5"),
    messages=[{"role": "user", "content": "hello"}],
    timeout=60,
)
print("Azure Chat Reply:", resp.choices[0].message.content)
