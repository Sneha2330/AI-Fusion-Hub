# audio_utils.py
import os
from azure_client import get_client

CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5")
WHISPER_DEPLOYMENT = os.getenv("AZURE_WHISPER_DEPLOYMENT", "whisper")

def transcribe_audio(file_path: str) -> str:
    try:
        if not file_path or not os.path.exists(file_path):
            return "No audio file received. Please record or upload a small WAV/MP3."
        client = get_client()
        with open(file_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=WHISPER_DEPLOYMENT,
                file=f
            )
        return resp.text or ""
    except Exception as e:
        return f"[Transcription error] {e}"

def summarize_text(text: str) -> str:
    try:
        if not text or not text.strip():
            return "Nothing to summarize. Transcribe first."
        client = get_client()
        resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "Summarize the transcript in clear bullet points."},
                {"role": "user", "content": text}
            ],
            max_completion_tokens=512,
            temperature=1
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[Summarization error] {e}"
