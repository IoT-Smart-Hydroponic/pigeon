import hmac
import hashlib
from fastapi import HTTPException
from .config import config


def verify_signature(payload: bytes, secret: str) -> None:
    signature = hmac.new(config.GITHUB_SECRET, payload, hashlib.sha256)
    expected = "sha256=" + signature.hexdigest()

    if not hmac.compare_digest(expected, secret):
        raise HTTPException(status_code=401, detail="Invalid signature")
