# whisper_test.py
import os
from pathlib import Path
from dotenv import load_dotenv
from azure_client import get_client

here = Path(__file__).resolve().parent
load_dotenv(dotenv_path=str(here / ".env"), override=True)

client = get_client()
whisper_dep = os.getenv("AZURE_WHISPER_DEPLOYMENT", "whisper")
print("Using Whisper deployment:", whisper_dep)

with open("sample.wav", "rb") as f:
    r = client.audio.transcriptions.create(
        model=whisper_dep,
        file=f,
        timeout=120
    )
print("TRANSCRIPT:", r.text)
