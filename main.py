# main.py -- FastAPI service Youtube shorts + Google Sheets logging + audio chunking

import os
import subprocess
import math
import tempfile
import json
import requests
import time
import logging
import argparse
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uuid
from datetime import datetime

# Google Sheets
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("youtube-shorts")

# ---------------------------------------------------------------------
# CLI arg for port (useful for local run)
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--port", type=int, help="Port to listen on (overrides PORT env)")
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

PORT = int(os.getenv("PORT", args.port or 8000))
OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # set this on the server
YTDLP_CMD = os.getenv("YTDLP_CMD", "yt-dlp")  # override if needed
FFPROBE_CMD = os.getenv("FFPROBE_CMD", "ffprobe")
FFMPEG_CMD = os.getenv("FFMPEG_CMD", "ffmpeg")

# chemin optionnel vers le fichier de cookies yt-dlp
YTDLP_COOKIES = os.getenv("YTDLP_COOKIES")

MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", 80000))

# limite de taille d'audio par requête OpenAI (en Mo) – marge de sécurité
MAX_AUDIO_MB = float(os.getenv("MAX_AUDIO_MB", "20"))

# Texte max stocké dans une cellule Google Sheets (pour éviter les cellules gigantesques)
MAX_SHEET_TEXT_CHARS = int(os.getenv("MAX_SHEET_TEXT_CHARS", 45000))

if OPENAI_KEY:
    logger.info("OPENAI_API_KEY present (len=%d)", len(OPENAI_KEY))
else:
    logger.warning("OPENAI_API_KEY not set — endpoints that use OpenAI will return 503 until configured.")

# ---------------------------------------------------------------------
# Google Sheets config
# ---------------------------------------------------------------------

GOOGLE_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_SERVICE_ACCOUNT_FILE",
    "/etc/secrets/credentials.json",  # à adapter si besoin
)

SPREADSHEET_ID = "1G7NKc76jtqCeCyTlaUZ1UArTLYKJjrxb563hlhyBOYo"
SHEET_NAME = "Feuille 1"

# colonnes (lettres)
COL_VIDEO_DURATION = "C"
COL_TRANSCRIPT_STATUS = "D"
COL_TRANSCRIPT_TEXT = "E"
COL_CLIP_PLAN_STATUS = "F"
COL_CLIP_PLAN_JSON = "G"
COL_TIMESTAMP = "H"

_sheets_service = None  # cache global


def get_sheets_service():
    global _sheets_service
    if _sheets_service is not None:
        return _sheets_service

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH, scopes=scopes)
    _sheets_service = build("sheets", "v4", credentials=creds)
    return _sheets_service


def create_request_row(request_id: str, youtube_url: str) -> int:
    """
    Crée une nouvelle ligne dans le sheet avec Request ID + URL + timestamp.
    Retourne l'index de ligne (2, 3, 4, ...)
    """
    service = get_sheets_service()
    sheet = service.spreadsheets()

    result = sheet.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET_NAME}!A:A"
    ).execute()
    values = result.get("values", [])
    row_index = len(values) + 1  # header = ligne 1

    now_iso = datetime.utcnow().isoformat()

    body = {
        "values": [[
            request_id,     # A: Request ID
            youtube_url,    # B: YouTube URL
            "",             # C: Video Duration
            "STARTED",      # D: Transcript Status
            "",             # E: Transcript Text
            "PENDING",      # F: Clip Plan Status
            "",             # G: Clip Plan JSON
            now_iso,        # H: Timestamp
        ]]
    }

    sheet.values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET_NAME}!A{row_index}:H{row_index}",
        valueInputOption="RAW",
        body=body,
    ).execute()

    return row_index


def update_single_cell(row_index: int, column_letter: str, value: str):
    service = get_sheets_service()
    sheet = service.spreadsheets()

    body = {"values": [[value]]}
    sheet.values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET_NAME}!{column_letter}{row_index}",
        valueInputOption="RAW",
        body=body,
    ).execute()


# ---------------------------------------------------------------------
# FastAPI app & models
# ---------------------------------------------------------------------

app = FastAPI(title="Youtube-shorts automation", version="0.4")


class ProcessRequest(BaseModel):
    youtube_url: str
    mode: str = "courte"   # "courte" or "longue"
    shorts_per_5min: float = 1.0   # multiplier


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def run_cmd_capture(cmd: List[str], timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Run a command and return dict {returncode, stdout, stderr}.
    Does not raise by itself (caller checks returncode).
    """
    logger.debug("run_cmd_capture: %s", cmd)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "returncode": p.returncode,
            "stdout": p.stdout or "",
            "stderr": p.stderr or "",
        }
    except subprocess.TimeoutExpired as e:
        logger.error("Command timeout: %s", cmd)
        return {"returncode": 124, "stdout": "", "stderr": f"timeout: {e}"}
    except Exception as e:
        logger.exception("Command failed: %s", cmd)
        return {"returncode": 1, "stdout": "", "stderr": str(e)}


def download_youtube_audio(url: str, outpath: str) -> float:
    """
    Download ONLY audio from youtube with yt-dlp to outpath.
    Returns duration (seconds) or raises HTTPException on failure.

    On se base sur l'existence + taille du fichier pour valider le download.
    Utilise éventuellement un fichier de cookies (YTDLP_COOKIES) si défini.
    """
    cmd: List[str] = [YTDLP_CMD, "-x", "--audio-format", "mp3"]

    if YTDLP_COOKIES:
        logger.info("Using cookies file: %s", YTDLP_COOKIES)
        cmd.extend(["--cookies", YTDLP_COOKIES])
    else:
        logger.warning("YTDLP_COOKIES not set, running yt-dlp without cookies")

    cmd.extend(["-o", outpath, url])

    logger.info("Downloading audio %s -> %s", url, outpath)
    res = run_cmd_capture(cmd, timeout=1800)

    logger.info("yt-dlp returncode=%s", res["returncode"])
    if res["stdout"]:
        logger.debug("yt-dlp stdout: %s", res["stdout"][:500])
    if res["stderr"]:
        logger.warning("yt-dlp stderr: %s", res["stderr"][:500])

    if not os.path.exists(outpath) or os.path.getsize(outpath) == 0:
        logger.error("yt-dlp did not produce a file at %s", outpath)
        snippet = (res["stderr"] or res["stdout"])[:300]
        raise HTTPException(
            status_code=500,
            detail=f"yt-dlp error: no output file. Logs: {snippet}"
        )

    # get duration using ffprobe
    p = run_cmd_capture([
        FFPROBE_CMD, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        outpath,
    ])
    if p["returncode"] != 0 or not p["stdout"].strip():
        logger.warning("ffprobe couldn't read duration: %s", p["stderr"][:500])
        return 0.0
    try:
        duration = float(p["stdout"].strip())
        logger.info("Downloaded audio duration: %.2f s", duration)
        return duration
    except Exception:
        logger.exception("Failed to parse duration from ffprobe output")
        return 0.0


def transcribe_with_openai(file_path: str) -> str:
    """
    Upload file to OpenAI Whisper endpoint and return transcript text.
    Raises HTTPException on failure, or returns empty string on unexpected response.
    """
    if not OPENAI_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured on server.")
    url = "https://api.openai.com/v1/audio/transcriptions"
    logger.info("Uploading file to OpenAI Whisper: %s", file_path)
    try:
        with open(file_path, "rb") as f:
            files = {"file": ("audio.mp3", f, "application/octet-stream")}
            data = {"model": "whisper-1"}
            headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
            r = requests.post(url, headers=headers, files=files, data=data, timeout=900)
        r.raise_for_status()
        resp_json = r.json()
        text = resp_json.get("text", "")
        if not text:
            logger.warning("OpenAI returned empty transcript")
        return text
    except requests.exceptions.RequestException as e:
        logger.exception("OpenAI transcription request failed")
        raise HTTPException(status_code=502, detail=f"OpenAI transcription error: {e}")


def transcribe_audio_in_chunks(
    file_path: str,
    duration_sec: float,
    row_index: Optional[int] = None
) -> Tuple[str, List[str]]:
    """
    Découpe l'audio en plusieurs fichiers si besoin, en fonction de la taille max (MAX_AUDIO_MB).
    Transcrit chaque chunk avec Whisper, concatène les textes et renvoie :
    - transcript complet
    - liste des chemins de fichiers temporaires à nettoyer.
    """
    size_bytes = os.path.getsize(file_path)
    logger.info("Audio size: %.2f MB, duration: %.2f s", size_bytes / (1024 * 1024), duration_sec)

    if duration_sec <= 0:
        # au pire : une seule requête
        if row_index:
            update_single_cell(row_index, COL_TRANSCRIPT_STATUS, "TRANSCRIBING 1/1 (no duration)")
        text = transcribe_with_openai(file_path)
        return text, []

    bytes_per_sec = size_bytes / duration_sec
    max_bytes = MAX_AUDIO_MB * 1024 * 1024
    approx_chunk_sec = max_bytes / bytes_per_sec if bytes_per_sec > 0 else duration_sec

    # minimum 60 secondes par chunk pour éviter 300 morceaux ridicules
    approx_chunk_sec = max(60.0, approx_chunk_sec)

    if duration_sec <= approx_chunk_sec * 1.05:
        # un seul chunk
        if row_index:
            update_single_cell(row_index, COL_TRANSCRIPT_STATUS, "TRANSCRIBING 1/1")
        text = transcribe_with_openai(file_path)
        return text, []

    # plusieurs chunks
    num_chunks = math.ceil(duration_sec / approx_chunk_sec)
    # on lisse un peu :
    chunk_duration = duration_sec / num_chunks

    logger.info(
        "Audio too large, splitting into %d chunks of ~%.2f s (max %.1f MB)",
        num_chunks, chunk_duration, MAX_AUDIO_MB
    )

    transcripts: List[str] = []
    tmp_files: List[str] = []

    base, ext = os.path.splitext(file_path)
    for i in range(num_chunks):
        start = i * chunk_duration
        this_dur = min(chunk_duration, duration_sec - start)
        part_path = f"{base}_part{i+1}{ext}"

        cmd = [
            FFMPEG_CMD,
            "-y",
            "-i", file_path,
            "-ss", str(start),
            "-t", str(this_dur),
            "-vn",        # pas de vidéo, au cas où
            "-acodec", "copy",
            part_path,
        ]
        logger.info("Creating chunk %d/%d: start=%.2f dur=%.2f -> %s",
                    i + 1, num_chunks, start, this_dur, part_path)
        res = run_cmd_capture(cmd, timeout=900)
        if res["returncode"] != 0:
            logger.error("ffmpeg chunk creation failed: %s", res["stderr"][:400])
            raise HTTPException(status_code=500, detail="Failed to split audio into chunks.")

        tmp_files.append(part_path)

        if row_index:
            update_single_cell(row_index, COL_TRANSCRIPT_STATUS, f"TRANSCRIBING {i+1}/{num_chunks}")

        part_text = transcribe_with_openai(part_path)
        transcripts.append(f"[PART {i+1}/{num_chunks} START={start:.2f}s]\n{part_text}\n")

    full_text = "\n".join(transcripts)
    return full_text, tmp_files


def ask_gpt_for_clips(transcript: str, desired_count: int, mode: str) -> List[Dict[str, Any]]:
    """
    Ask OpenAI chat completions to propose clips. Returns parsed JSON list.
    """
    if not OPENAI_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured on server.")
    safe_transcript = transcript[:MAX_TRANSCRIPT_CHARS]
    prompt = (
        "You are given a transcription of a video. Return a JSON array of the best "
        f"{desired_count} clips to publish as short videos. Mode = {mode}. "
        "If mode is 'courte' choose clips ~30s (allow 15-45s). If mode is 'longue' choose clips between 61 and 105 seconds. "
        "For each clip return: {index, start (seconds), end (seconds), duration, title (5-8 words), excerpt (text excerpt to show as subtitle)}. "
        "Return ONLY a JSON array (no extra comments)."
        "\n\nTRANSCRIPT:\n" + safe_transcript
    )
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1500
    }
    logger.info("Requesting GPT for %d clips (mode=%s)", desired_count, mode)
    try:
        r = requests.post(url, headers=headers, json=body, timeout=120)
        r.raise_for_status()
        resp = r.json()
        txt = resp["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        logger.exception("OpenAI chat completion failed")
        raise HTTPException(status_code=502, detail=f"OpenAI completion error: {e}")

    try:
        data = json.loads(txt)
        if not isinstance(data, list):
            raise ValueError("expected list")
        return data
    except Exception:
        import re
        m = re.search(r'(\[.*\])', txt, re.S)
        if m:
            try:
                data = json.loads(m.group(1))
                return data
            except Exception:
                logger.exception("Failed to parse JSON from GPT fallback")
        logger.error("GPT response parsing failed. Raw start: %s", txt[:500])
        raise HTTPException(status_code=502, detail="Failed to parse GPT response as JSON.")


# ---------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------

@app.post("/process")
def process(req: ProcessRequest, background: BackgroundTasks):
    """
    Synchronous processing: download audio, transcribe (avec chunking),
    ask GPT, return clips. Log dans Google Sheets.
    Les fichiers temporaires sont nettoyés en background.
    """
    logger.info(
        "Process request: url=%s mode=%s shorts_per_5min=%s",
        req.youtube_url, req.mode, req.shorts_per_5min
    )

    # request id + ligne sheet
    request_id = str(uuid.uuid4())
    row_index = create_request_row(request_id, req.youtube_url)

    tmpdir = tempfile.mkdtemp(prefix="ytshorts_")
    audio_path = os.path.join(tmpdir, "audio.mp3")
    temp_files: List[str] = [audio_path]

    duration = 0.0
    transcript = ""
    clips: List[Dict[str, Any]] = []

    try:
        # 1) download audio
        duration = download_youtube_audio(req.youtube_url, audio_path)
        update_single_cell(row_index, COL_VIDEO_DURATION, str(duration))
        update_single_cell(row_index, COL_TRANSCRIPT_STATUS, "AUDIO_DOWNLOADED")

        # 2) compute number of shorts
        shorts_count = max(1, math.ceil((duration / 60.0) / 5.0 * float(req.shorts_per_5min)))

        # 3) transcribe (chunked if needed)
        transcript, chunk_files = transcribe_audio_in_chunks(audio_path, duration, row_index=row_index)
        temp_files.extend(chunk_files)
        update_single_cell(row_index, COL_TRANSCRIPT_STATUS, "TRANSCRIBED_OK")
        # on tronque pour éviter cellule énorme
        update_single_cell(row_index, COL_TRANSCRIPT_TEXT, transcript[:MAX_SHEET_TEXT_CHARS])

        # 4) ask GPT for clips
        update_single_cell(row_index, COL_CLIP_PLAN_STATUS, "CLIP_PLAN_RUNNING")
        clips = ask_gpt_for_clips(transcript, shorts_count, req.mode)

        # 5) normalize/clamp durations
        out_clips = []
        for i, c in enumerate(clips):
            try:
                s = float(c.get("start", 0))
            except Exception:
                s = 0.0
            try:
                e = float(c.get("end", s + (c.get("duration") or 30)))
            except Exception:
                e = s + 30.0
            dur = e - s
            if req.mode == "courte":
                if dur < 15:
                    e = s + 15
                if dur > 45:
                    e = s + 45
            else:
                if dur < 61:
                    e = s + 61
                if dur > 105:
                    e = s + 105
            out_clips.append({
                "index": int(c.get("index", i + 1)),
                "start": round(max(0.0, s), 2),
                "end": round(max(s + 0.01, e), 2),
                "duration": round(max(0.0, e - s), 2),
                "title": c.get("title", "") or "",
                "excerpt": c.get("excerpt", "") or ""
            })

        clips_json = json.dumps(out_clips)
        update_single_cell(row_index, COL_CLIP_PLAN_STATUS, "CLIP_PLAN_OK")
        update_single_cell(row_index, COL_CLIP_PLAN_JSON, clips_json[:MAX_SHEET_TEXT_CHARS])

    except HTTPException as e:
        # log erreur dans le sheet
        msg = f"ERROR {e.status_code}: {e.detail}"
        logger.error("Process failed: %s", msg)
        try:
            # si échec avant transcription
            update_single_cell(row_index, COL_TRANSCRIPT_STATUS, msg[:500])
            update_single_cell(row_index, COL_CLIP_PLAN_STATUS, "ERROR")
        except Exception:
            logger.exception("Failed to update Google Sheet with error")
        raise
    except Exception as e:
        logger.exception("Unexpected error in /process")
        msg = f"UNEXPECTED_ERROR: {e}"
        try:
            update_single_cell(row_index, COL_TRANSCRIPT_STATUS, msg[:500])
            update_single_cell(row_index, COL_CLIP_PLAN_STATUS, "ERROR")
        except Exception:
            logger.exception("Failed to update Google Sheet with unexpected error")
        raise HTTPException(status_code=500, detail=str(e))

    # cleanup en background
    def cleanup(paths: List[str], tmp: str):
        try:
            for p in paths:
                if os.path.exists(p):
                    os.remove(p)
            if os.path.isdir(tmp):
                os.rmdir(tmp)
            logger.info("Cleaned up %s", tmp)
        except Exception:
            logger.exception("Cleanup failed")

    background.add_task(cleanup, temp_files, tmpdir)

    return {
        "request_id": request_id,
        "video_url": req.youtube_url,
        "duration_seconds": duration,
        "shorts": out_clips,
    }


@app.get("/")
def root():
    return {"status": "ok", "service": "youtube-shorts", "openai": bool(OPENAI_KEY), "port": PORT}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn on 0.0.0.0:%d", PORT)
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
