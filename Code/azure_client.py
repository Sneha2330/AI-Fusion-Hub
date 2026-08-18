# azure_client.py
import os
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
from openai import AzureOpenAI
import httpx


def _load_env_from_here():
    """Load .env from the same directory as this file (not from CWD)."""
    here = Path(__file__).resolve().parent
    env_path = here / ".env"

    # 1) Try standard loader (expects UTF-8 .env)
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
    else:
        return

    # 2) If still missing (edge cases like weird encodings), parse manually as UTF-8
    if not os.getenv("AZURE_OPENAI_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
        try:
            vals = dotenv_values(dotenv_path=str(env_path), encoding="utf-8")
            for k, v in vals.items():
                if v and os.getenv(k) is None:
                    os.environ[k] = v
        except Exception:
            pass


_load_env_from_here()


def get_client():
    """
    Returns a configured AzureOpenAI client.
    Reads:
      - AZURE_OPENAI_KEY
      - AZURE_OPENAI_ENDPOINT
    Accepts alternates:
      - AZURE_OPENAI_API_KEY
      - OPENAI_API_BASE

    Honors optional proxy/CA via environment variables (no code params needed):
      - HTTP_PROXY / HTTPS_PROXY
      - SSL_CERT_FILE / REQUESTS_CA_BUNDLE
    """
    key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")

    if not key or not endpoint:
        raise RuntimeError(f"Missing env vars. KEY={key}, ENDPOINT={endpoint}")

    # Respect proxies/CA/timeouts via env; do not pass 'proxies=' (to support older httpx)
    ca_bundle = os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE") or True
    timeout = httpx.Timeout(connect=30.0, read=60.0, write=60.0, pool=60.0)

    http_client = httpx.Client(
        verify=ca_bundle,
        timeout=timeout,
        # NOTE: no 'proxies=' here to avoid TypeError on older httpx
    )

    return AzureOpenAI(
        api_key=key,
        azure_endpoint=endpoint,
        api_version="2024-12-01-preview",
        http_client=http_client,
        timeout=60,  # per-request default fallback
    )
