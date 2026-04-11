"""
youtube_uploader.py - Upload processed videos to YouTube as unlisted.

Uses OAuth2 installed-app flow with client_secrets.json bundled in the build.
Token is cached at ~/.config/eve-trimmer/youtube_token.json and auto-refreshed.
"""

import os
import sys
import threading
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


def get_credentials(cancel_event=None) -> Credentials:
    """Load cached credentials or run OAuth2 browser flow.

    If cancel_event (threading.Event) is provided and set while the browser
    OAuth flow is waiting, raises RuntimeError so the caller can clean up.
    """
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

            result: list = []
            exc_holder: list = []

            def _run_flow():
                try:
                    result.append(flow.run_local_server(
                        port=0,
                        success_message="""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Authorization Complete</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #0f0f0f;
      color: #e8e8e8;
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100vh;
      margin: 0;
    }
    .card {
      background: #1a1a1a;
      border: 1px solid #333;
      border-radius: 12px;
      padding: 40px 48px;
      text-align: center;
      max-width: 400px;
    }
    .check {
      font-size: 48px;
      margin-bottom: 16px;
    }
    h1 {
      font-size: 22px;
      font-weight: 600;
      margin: 0 0 8px;
    }
    p {
      color: #999;
      font-size: 14px;
      margin: 0;
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="check">&#10003;</div>
    <h1>YouTube authorized</h1>
    <p>You can close this tab and return to Eve Trimmer.</p>
  </div>
</body>
</html>""",
                    ))
                except Exception as e:
                    exc_holder.append(e)

            oauth_thread = threading.Thread(target=_run_flow, daemon=True)
            oauth_thread.start()

            while oauth_thread.is_alive():
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("OAuth cancelled by user")
                oauth_thread.join(timeout=0.1)

            if exc_holder:
                raise exc_holder[0]
            creds = result[0]

        _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        _TOKEN_PATH.write_text(creds.to_json())
    return creds


def upload(
    video_path: str,
    title: str,
    description: str,
    status_callback=None,
    cancel_event=None,
) -> str:
    """
    Upload video_path to YouTube as an unlisted video.

    Args:
        video_path: Path to the video file.
        title: YouTube video title.
        description: Video description (YouTube chapter timestamps).
        status_callback: Optional callable(pct: int) called with upload progress 0-100.
        cancel_event: Optional threading.Event; if set, upload is aborted.

    Returns:
        YouTube URL of the uploaded video (https://youtu.be/<id>).
    """
    creds = get_credentials(cancel_event=cancel_event)
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
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Upload cancelled by user")
        status, response = request.next_chunk()
        if status and status_callback:
            status_callback(int(status.progress() * 100))

    if status_callback:
        status_callback(100)

    return f"https://youtu.be/{response['id']}"
