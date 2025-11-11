# main.py -- FastAPI service minimal
import os, subprocess, math, tempfile, json, requests, time
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Set OPENAI_API_KEY")

app = FastAPI()

class ProcessRequest(BaseModel):
    youtube_url: str
    mode: str = "courte"   # "courte" or "longue"
    shorts_per_5min: float = 1.0   # 1 short per 5min by default (can be changed)

def download_youtube(url, outpath):
    cmd = ["yt-dlp", "-f", "best", "-o", outpath, url]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # get duration using ffprobe
    p = subprocess.run(
        ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", outpath],
        capture_output=True, text=True
    )
    duration = float(p.stdout.strip())
    return duration

def transcribe_with_openai(file_path):
    url = "https://api.openai.com/v1/audio/transcriptions"
    with open(file_path, "rb") as f:
        files = {"file": ("video.mp4", f, "application/octet-stream")}
        data = {"model":"whisper-1"}
        headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
        r = requests.post(url, headers=headers, files=files, data=data, timeout=600)
    r.raise_for_status()
    return r.json().get("text","")

def ask_gpt_for_clips(transcript, desired_count, mode):
    # simple prompt to return JSON list of clips with start/end in seconds and short title + excerpt
    prompt = (
        "You are given a transcription of a video. Return a JSON array of the best "
        f"{desired_count} clips to publish as short videos. Mode = {mode}. "
        "If mode is 'courte' choose clips ~30s (allow 15-45s). If mode is 'longue' choose clips between 61 and 105 seconds. "
        "For each clip return: {index, start (seconds), end (seconds), duration, title (5-8 words), excerpt (text excerpt to show as subtitle)}. "
        "Return only valid JSON array."
        "\n\nTRANSCRIPT:\n" + transcript[:80000]  # safety cap
    )
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type":"application/json"}
    body = {
        "model":"gpt-4o-mini", 
        "messages":[{"role":"user","content":prompt}],
        "temperature":0.2,
        "max_tokens":1500
    }
    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    resp = r.json()
    txt = resp["choices"][0]["message"]["content"]
    # try parse JSON
    try:
        data = json.loads(txt)
        return data
    except Exception:
        # fallback: try to extract JSON substring
        import re
        m = re.search(r'(\[.*\])', txt, re.S)
        if m:
            return json.loads(m.group(1))
        raise

@app.post("/process")
def process(req: ProcessRequest, background: BackgroundTasks):
    # 1) prepare temp file
    tmpdir = tempfile.mkdtemp()
    outpath = tmpdir + "/video.mp4"
    # 2) download (blocking)
    download_youtube(req.youtube_url, outpath)
    duration = None
    try:
        p = subprocess.run(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", outpath],
            capture_output=True, text=True
        )
        duration = float(p.stdout.strip())
    except:
        duration = 0.0
    # 3) compute number of shorts
    shorts_count = max(1, math.ceil((duration/60) / 5 * req.shorts_per_5min))
    # 4) transcribe via OpenAI Whisper
    transcript = transcribe_with_openai(outpath)
    # 5) ask GPT for clips
    clips = ask_gpt_for_clips(transcript, shorts_count, req.mode)
    # ensure each clip obeys mode duration constraints (clamp)
    out_clips = []
    for i,c in enumerate(clips):
        s = float(c.get("start",0))
        e = float(c.get("end", s + (c.get("duration") or 30)))
        dur = e - s
        if req.mode=="courte":
            # clamp 15-45
            if dur < 15: e = s + 15
            if dur > 45: e = s + 45
        else:
            if dur < 61: e = s + 61
            if dur > 105: e = s + 105
        out_clips.append({
            "index": i+1,
            "start": round(s,2),
            "end": round(e,2),
            "duration": round(e-s,2),
            "title": c.get("title",""),
            "excerpt": c.get("excerpt","")
        })
    # 6) return JSON result (client will get immediately)
    result = {"video_url": req.youtube_url, "duration_seconds": duration, "shorts": out_clips}
    # cleanup - delete file async
    def cleanup(path=outpath):
        try:
            os.remove(path)
            os.rmdir(tmpdir)
        except: pass
    background.add_task(cleanup)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
