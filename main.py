# main.py -- FastAPI service minimal (corrected)
import os
import subprocess
import math
import tempfile
import json
import requests
import time
import logging
import argparse
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("youtube-shorts")

# CLI arg for port (useful for local run)
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--port", type=int, help="Port to listen on (overrides PORT env)")
args, _ = parser.parse_known_args()

# config
PORT = int(os.getenv("PORT", args.port or 8000))
OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # set this on the server
YTDLP_CMD = os.getenv("YTDLP_CMD", "yt-dlp")  # override if needed
FFPROBE_CMD = os.getenv("FFPROBE_CMD", "ffprobe")
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", 80000))

if OPENAI_KEY:
    logger.info("OPENAI_API_KEY present (len=%d)", len(OPENAI_KEY))
else:
    logger.warning("OPENAI_API_KEY not set — endpoints that use OpenAI will return 503 until configured.")

app = FastAPI(title="Youtube-shorts automation", version="0.2")

class ProcessRequest(BaseModel):
    youtube_url: str
    mode: str = "courte"   # "courte" or "longue"
    shorts_per_5min: float = 1.0   # multiplier

def run_cmd_capture(cmd: List[str], timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Run a command and return dict {returncode, stdout, stderr}.
    Does not raise by itself (caller checks returncode).
    """
    logger.debug("run_cmd_capture: %s", cmd)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"returncode": p.returncode, "stdout": p.stdout or "", "stderr": p.stderr or ""}
    except subprocess.TimeoutExpired as e:
        logger.error("Command timeout: %s", cmd)
        return {"returncode": 124, "stdout": "", "stderr": f"timeout: {e}"}
    except Exception as e:
        logger.exception("Command failed: %s", cmd)
        return {"returncode": 1, "stdout": "", "stderr": str(e)}

def download_youtube(url: str, outpath: str) -> float:
    """
    Download youtube video with yt-dlp to outpath.
    Returns duration (seconds) or raises HTTPException on failure.

    STRATÉGIE SIMPLE :
    - On lance yt-dlp.
    - On ignore totalement le code de retour.
    - Si le fichier de sortie n'existe pas ou fait 0 octet => erreur.
    - Sinon on continue avec ffprobe pour la durée.
    """
    cmd = [YTDLP_CMD, "-f", "best", "-o", outpath, url]
    logger.info("Downloading %s -> %s", url, outpath)

    res = run_cmd_capture(cmd, timeout=900)
    stderr_short = (res.get("stderr") or "")[:500]

    # 1) Vérifier si le fichier a été créé
    if not os.path.exists(outpath) or os.path.getsize(outpath) == 0:
        logger.error(
            "yt-dlp failed, output file missing or empty. rc=%s, stderr=%s",
            res.get("returncode"),
            stderr_short,
        )
        raise HTTPException(
            status_code=500,
            detail=f"yt-dlp error: {stderr_short}",
        )

    # 2) Log si rc != 0 (warning, mais on continue)
    if res.get("returncode") != 0:
        logger.warning(
            "yt-dlp returned non-zero code (%s) but file exists (%s bytes). "
            "Continuing. stderr (truncated): %s",
            res.get("returncode"),
            os.path.getsize(outpath),
            stderr_short,
        )

    # 3) Récupérer la durée avec ffprobe
    p = run_cmd_capture([
        FFPROBE_CMD, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        outpath,
    ])
    if p["returncode"] != 0 or not p["stdout"].strip():
        logger.warning(
            "ffprobe couldn't read duration: %s",
            (p.get("stderr") or "")[:500],
        )
        return 0.0

    try:
        duration = float(p["stdout"].strip())
        logger.info("Downloaded duration: %.2f s", duration)
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
            files = {"file": ("video.mp4", f, "application/octet-stream")}
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

def ask_gpt_for_clips(transcript: str, desired_count: int, mode: str) -> List[Dict[str, Any]]:
    """
    Ask OpenAI chat completions to propose clips. Returns parsed JSON list.
    """
    if not OPENAI_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured on server.")
    # truncate transcript safely
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
        "max_tokens": 1500,
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
    # parse JSON with fallback
    try:
        data = json.loads(txt)
        if not isinstance(data, list):
            raise ValueError("expected list")
        return data
    except Exception:
        import re
        m = re.search(r"(\[.*\])", txt, re.S)
        if m:
            try:
                data = json.loads(m.group(1))
                return data
            except Exception:
                logger.exception("Failed to parse JSON from GPT fallback")
        logger.error("GPT response parsing failed. Raw start: %s", txt[:500])
        raise HTTPException(status_code=502, detail="Failed to parse GPT response as JSON.")

@app.post("/process")
def process(req: ProcessRequest, background: BackgroundTasks):
    """
    Synchronous processing: download, transcribe, ask GPT, return clips.
    The heavy temp files are scheduled for cleanup in background.
    """
    logger.info(
        "Process request: url=%s mode=%s shorts_per_5min=%s",
        req.youtube_url,
        req.mode,
        req.shorts_per_5min,
    )
    tmpdir = tempfile.mkdtemp(prefix="ytshorts_")
    outpath = os.path.join(tmpdir, "video.mp4")
    # 1) download
    duration = download_youtube(req.youtube_url, outpath)
    # 2) compute number of shorts
    shorts_count = max(1, math.ceil((duration / 60.0) / 5.0 * float(req.shorts_per_5min)))
    # 3) transcribe
    transcript = transcribe_with_openai(outpath)
    # 4) ask GPT for clips
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
        out_clips.append(
            {
                "index": int(c.get("index", i + 1)),
                "start": round(max(0.0, s), 2),
                "end": round(max(s + 0.01, e), 2),
                "duration": round(max(0.0, e - s), 2),
                "title": c.get("title", "") or "",
                "excerpt": c.get("excerpt", "") or "",
            }
        )
    # schedule cleanup
    def cleanup(path=outpath, tmp=tmpdir):
        try:
            if os.path.exists(path):
                os.remove(path)
            if os.path.isdir(tmp):
                os.rmdir(tmp)
            logger.info("Cleaned up %s", tmp)
        except Exception:
            logger.exception("Cleanup failed")

    background.add_task(cleanup)
    return {"video_url": req.youtube_url, "duration_seconds": duration, "shorts": out_clips}

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
