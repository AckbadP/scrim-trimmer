"""
youtube_uploader.py - Upload processed videos to YouTube as unlisted.

Uses OAuth2 installed-app flow with client_secrets.json bundled in the build.
Token is cached at ~/.config/eve-trimmer/youtube_token.json and auto-refreshed.
"""

import os
import sys
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
_TOKEN_PATH = Path.home() / ".config" / "eve-trimmer" / "youtube_token.json"


def _secrets_path() -> str:
    """Return path to bundled client_secrets.json."""
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "client_secrets.json")


def get_credentials() -> Credentials:
    """Load cached credentials or run OAuth2 browser flow."""
    creds = None
    if _TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            secrets = _secrets_path()
            if not os.path.exists(secrets):
                raise FileNotFoundError(
                    "client_secrets.json not found — "
                    "YouTube upload is unavailable in this build."
                )
            flow = InstalledAppFlow.from_client_secrets_file(secrets, SCOPES)
            creds = flow.run_local_server(port=0)
        _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TOKEN_PATH.write_text(creds.to_json())
    return creds


def upload(
    video_path: str,
    title: str,
    description: str,
    status_callback=None,
) -> str:
    """
    Upload video_path to YouTube as an unlisted video.

    Args:
        video_path: Path to the video file.
        title: YouTube video title.
        description: Video description (YouTube chapter timestamps).
        status_callback: Optional callable(pct: int) called with upload progress 0-100.

    Returns:
        YouTube URL of the uploaded video (https://youtu.be/<id>).
    """
    creds = get_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": "20",  # Gaming
        },
        "status": {
            "privacyStatus": "unlisted",
        },
    }

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        resumable=True,
        chunksize=4 * 1024 * 1024,
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status and status_callback:
            status_callback(int(status.progress() * 100))

    if status_callback:
        status_callback(100)

    return f"https://youtu.be/{response['id']}"
