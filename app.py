# app.py
# ------------------------------------------------------------------------------------
# Hume (Expression Measurement – Prosody) + Omi (Imports + Notification)
# Deterministic “Vocal Tone Report” — NO LLM REQUIRED
#
# IMPORTANT (per your note): Use OMI_APP_SECRET for ALL Omi calls (Imports + Notification).
#
# ENV (Render → Environment):
#   OMI_APP_ID       = your Omi Integration ID
#   OMI_APP_SECRET   = your Omi App Secret    (used for BOTH Imports + Notification)
#   HUME_API_KEY     = your Hume API key
#
# Start command (Render → Start Command):
#   uvicorn app:app --host 0.0.0.0 --port $PORT
#
# Build command:
#   pip install -r requirements.txt
#
# requirements.txt (example):
#   fastapi>=0.115
#   uvicorn[standard]>=0.30
#   httpx>=0.27
#   pydantic>=2.8
#   hume>=0.13,<0.14
#   python-dotenv>=1.0
#
# Recommended runtime.txt (Render → root of repo) to avoid Python 3.13 issues:
#   python-3.12.6
# ------------------------------------------------------------------------------------

import os
import re
import io
import wave
import json
import time
import tempfile
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Query
from pydantic import BaseModel, Field

# Hume SDK (modern async client & Expression Measurement stream)
from hume.client import AsyncHumeClient
from hume.expression_measurement.stream import Config as EMConfig
from hume.expression_measurement.stream.socket_client import StreamConnectOptions

PID = os.getpid()

# =========================
# ENV / CONSTANTS
# =========================
OMI_APP_ID = os.environ.get("OMI_APP_ID")
OMI_APP_SECRET = os.environ.get("OMI_APP_SECRET")  # <- used for ALL Omi endpoints (per your instruction)
HUME_API_KEY = os.environ.get("HUME_API_KEY")

IDLE_TIMEOUT_SEC = int(os.environ.get("IDLE_TIMEOUT_SEC", "180"))  # finalize after inactivity
MAX_CHUNK_SEC = float(os.environ.get("MAX_CHUNK_SEC", "4.8"))      # keep chunks <5s for smoother streaming

# Optional text markers (if you later reuse transcript logic)
START_RE = re.compile(r"\bconversation\s*starts\b", re.I)
END_RE   = re.compile(r"\bconversa(?:i?t)ion\s*end(?:s)?\b", re.I)

# =========================
# FastAPI app & clients
# =========================
http_client: Optional[httpx.AsyncClient] = None
hume_client: Optional[AsyncHumeClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, hume_client
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🚀 Server starting...")
    http_client = httpx.AsyncClient(timeout=45.0)
    if HUME_API_KEY:
        hume_client = AsyncHumeClient(api_key=HUME_API_KEY)
    else:
        hume_client = None
    try:
        yield
    finally:
        if http_client:
            await http_client.aclose()
        print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 👋 Server shutting down...")

app = FastAPI(title="Vocal Tone (Hume Prosody x Omi, deterministic)", lifespan=lifespan)

# =========================
# Omi helpers (ALL use OMI_APP_SECRET per your request)
# =========================
async def omi_create_conversation(uid_or_session: str, text: str):
    if not (OMI_APP_ID and OMI_APP_SECRET and http_client):
        print("❌ Omi create_conversation: missing creds/client")
        return
    url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/user/conversations"
    headers = {"Authorization": f"Bearer {OMI_APP_SECRET}", "Content-Type": "application/json"}
    params = {"uid": uid_or_session}
    payload = {
        "text": text,
        "text_source": "other_text",
        "text_source_spec": "vocal_tone_report",
        "language": "en",
    }
    r = await http_client.post(url, headers=headers, params=params, json=payload)
    if r.status_code // 100 != 2:
        print(f"❌ Omi create_conversation {r.status_code} {r.text[:300]}")

async def omi_send_notification(uid_or_session: str, title: str, body: str):
    if not (OMI_APP_ID and OMI_APP_SECRET and http_client):
        print("❌ Omi notification: missing creds/client")
        return
    url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/notification"
    headers = {"Authorization": f"Bearer {OMI_APP_SECRET}", "Content-Type": "application/json", "Content-Length": "0"}
    # keep body relatively short to avoid device truncation
    params = {"uid": uid_or_session, "message": f"{title}: {body}"[:1800]}
    try:
        r = await http_client.post(url, headers=headers, params=params, timeout=15.0)
        if r.status_code // 100 != 2:
            print(f"❌ Omi notification {r.status_code} {r.text[:300]}")
    except Exception as e:
        print(f"❌ Omi notification error: {e}")

async def omi_save_memory(uid_or_session: str, full_text: str):
    if not (OMI_APP_ID and OMI_APP_SECRET and http_client):
        print("❌ Omi memory: missing creds/client")
        return
    # Ensure full text saved as conversation first
    await omi_create_conversation(uid_or_session, full_text)

    # Then create a short searchable memory
    url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/user/memories"
    headers = {"Authorization": f"Bearer {OMI_APP_SECRET}", "Content-Type": "application/json"}
    params = {"uid": uid_or_session}
    first_line = (full_text.splitlines() or [""])[0][:300]
    payload = {
        "text": full_text,
        "text_source": "other",
        "text_source_spec": "vocal_tone",
        "memories": [
            {"content": first_line, "tags": ["vocal_tone", "hume", "prosody", "report"]}
        ],
    }
    r = await http_client.post(url, headers=headers, params=params, json=payload)
    if r.status_code // 100 != 2:
        print(f"❌ Omi memory {r.status_code} {r.text[:300]}")

# =========================
# Aggregation & deterministic scoring
# =========================
class EmotionAgg:
    """
    Aggregates Hume prosody outputs across slices.
    Stores totals and peak events for reporting.
    """
    def __init__(self):
        self.total_dur = 0.0
        self.emotion_sum: Dict[str, float] = {}
        self.peak_events: List[Tuple[float, str, float]] = []  # (timestamp_s, name, score)

    def add_predictions(self, result: Dict[str, Any], slice_dur: float, t0: float):
        """
        result: Hume EM stream result object (dict-like)
        Expect: result.prosody.predictions[*].time{begin,end}, .emotions[*]{name,score}
        """
        prosody = result.get("prosody") or result.get("models", {}).get("prosody")
        if not prosody:
            return
        preds = prosody.get("predictions", [])
        for p in preds:
            time_obj = p.get("time", {})
            begin = float(time_obj.get("begin", 0.0))
            emotions = p.get("emotions", [])
            for e in emotions:
                name = str(e.get("name", "unknown")).strip()
                score = float(e.get("score", 0.0))
                self.emotion_sum[name] = self.emotion_sum.get(name, 0.0) + score
                if score >= 0.75:  # keep notable peaks
                    self.peak_events.append((t0 + begin, name, score))
        self.total_dur += max(slice_dur, 1e-6)

POSITIVE = {
    "Admiration","Amusement","Awe","Calmness","Contentment","Interest",
    "Joy","Relief","Triumph","Tenderness"
}
TENSION  = {
    "Annoyance","Anxiety","Distress","Frustration","Irritation",
    "Sadness","Disappointment","Contempt","Anger","Doubt"
}
ENERGY   = {"Excitement","Elation","Determination","Surprise"}
ENGAGE   = {"Interest","Curiosity","Concentration","Enthusiasm","Attentiveness"}

def _avg_over(agg: EmotionAgg, names: set) -> float:
    if agg.total_dur <= 0:
        return 0.0
    s = sum(agg.emotion_sum.get(n, 0.0) for n in names)
    return 100.0 * (s / max(agg.total_dur, 1e-6))

def compute_metrics(agg: EmotionAgg) -> dict:
    warmth   = max(0, min(100, int(_avg_over(agg, POSITIVE))))
    tension  = max(0, min(100, int(_avg_over(agg, TENSION))))
    energy   = max(0, min(100, int(_avg_over(agg, ENERGY))))
    engage   = max(0, min(100, int(_avg_over(agg, ENGAGE))))
    # crude empathy proxy: co-occurrence of tension & positive peaks in session
    empathy  = 50
    if agg.peak_events:
        has_tension  = any(n in TENSION for _, n, _ in agg.peak_events)
        has_positive = any(n in POSITIVE for _, n, _ in agg.peak_events)
        empathy = 70 if (has_tension and has_positive) else (60 if has_positive else 50)
    vocal_conf = max(0, min(100, int(0.6*energy + 0.4*(100 - tension))))
    vocal_tone = max(0, min(100, int(0.5*warmth + 0.3*vocal_conf + 0.2*engage - 0.1*tension)))
    ai_conf    = min(95, int(20 + 75 * min(1.0, agg.total_dur / 60.0)))  # more voiced time → higher
    return {
        "warmth": warmth, "tension": tension, "energy": energy, "engage": engage,
        "vocal_conf": vocal_conf, "vocal_tone": vocal_tone, "ai_conf": ai_conf,
    }

def build_report(agg: EmotionAgg, title_hint: str = "Vocal Tone Session") -> str:
    m = compute_metrics(agg)
    # top 2 emotions overall
    top = sorted(agg.emotion_sum.items(), key=lambda x: x[1], reverse=True)[:2]
    top_lines = [f"• Strong **{n}** signal" for n, _ in top] or ["• Clear articulation", "• Stable prosody"]
    # two notable peaks (highest scores)
    peaks = sorted(agg.peak_events, key=lambda x: x[2], reverse=True)[:2]
    peak_lines = []
    for ts, name, score in peaks:
        peak_lines.append(f'• ~{int(ts)}s: high **{name}** ({score:.2f}) — notable moment')

    summary = (
        "Overall delivery leaned " +
        ("warm and engaged." if m["warmth"] >= 60 else "neutral.") +
        (" Dynamic energy helped confidence." if m["vocal_conf"] >= 60 else " Consider adding more dynamic range.")
    )

    return (
        f'Vocal Tone: {m["vocal_tone"]} — "{title_hint}"\n'
        f'AI Confidence: {m["ai_conf"]}\n\n'
        f'{summary}\n\n'
        f'✅ Highlights:\n'
        f'{top_lines[0]}\n'
        f'{(top_lines[1] if len(top_lines)>1 else "• Consistent pacing")}\n\n'
        f'💡 Try:\n'
        f'• If tension rises, slow delivery and add brief pauses.\n'
        f'• Vary pitch/emphasis to avoid flat delivery.\n\n'
        f'Breakdown:\n'
        f'• Overall Positivity: {m["warmth"]} (positivity in prosody)\n'
        f'• Vocal Confidence: {m["vocal_conf"]} (energy vs. strain)\n'
        f'• Engagement Level: {m["engage"]} (interest/attention cues)\n'
        f'• Empathy: {max(0,min(100,int((m["warmth"]+100-m["tension"])/2)))} (approximate)\n'
        f'---\n'
        f'✨ Emotional Moments:\n'
        f'{(peak_lines[0] if peak_lines else "• Moments were evenly expressed")}\n'
        f'{(peak_lines[1] if len(peak_lines)>1 else "")}\n'
        f'\n'
        f'*Report derived from vocal expressions only; labels reflect perceived vocal tone, not inner feelings.*'
    )

# =========================
# In-memory per-session state
# =========================
class VoiceState:
    def __init__(self, key: str):
        self.key = key  # uid or session_id
        self.active = False
        self.started_at = 0.0
        self.last_wall_ts = 0.0
        self.last_audio_ts = 0.0
        self.agg = EmotionAgg()
        self.lock = asyncio.Lock()

    def touch(self):
        self.last_wall_ts = time.time()

STATES: Dict[str, VoiceState] = {}
STATES_LOCK = asyncio.Lock()

async def _get_state(key: str) -> VoiceState:
    async with STATES_LOCK:
        if key not in STATES:
            STATES[key] = VoiceState(key)
        return STATES[key]

def _effective_id(uid: Optional[str], session_id: Optional[str], headers) -> Optional[str]:
    # Prefer uid, fallback to session_id, then header hints
    return uid or session_id or headers.get("X-Session-ID") or headers.get("X-Omi-Uid")

# =========================
# Hume streaming utilities
# =========================
def _bytes_to_wav_path(raw: bytes, sample_rate: int) -> str:
    """
    Wrap raw PCM16 mono bytes with a WAV header and write to a temp file, returning the path.
    This makes the chunk a valid audio file for Hume's send_file().
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # PCM16
        wf.setframerate(sample_rate)
        wf.writeframes(raw)
    return tmp_path

async def hume_measure_bytes(audio_bytes: bytes, sample_rate: int) -> Optional[Dict[str, Any]]:
    """
    Use Hume Expression Measurement (Prosody) stream for a short chunk.
    """
    if not hume_client:
        print("❌ Hume client not configured")
        return None
    model_config = EMConfig(prosody={})  # enable prosody only
    options = StreamConnectOptions(config=model_config)
    wav_path = _bytes_to_wav_path(audio_bytes, sample_rate)
    try:
        async with hume_client.expression_measurement.stream.connect(options=options) as socket:
            result = await socket.send_file(wav_path)
            # `result` is an EM result object (dict-like); return as dict
            return result
    except Exception as e:
        print(f"❌ Hume stream error: {e}")
        return None
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass

# =========================
# Finalization
# =========================
async def finalize_and_report(st: VoiceState, title_hint="Vocal Tone Session") -> Dict[str, Any]:
    key = st.key
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🟥 Finalizing id={key}")

    report_text = build_report(st.agg, title_hint=title_hint)

    # Push to Omi (ALL use OMI_APP_SECRET per your instruction)
    await omi_send_notification(key, title="Your Vocal Tone Report", body=report_text[:800])
    await omi_create_conversation(key, report_text)
    await omi_save_memory(key, report_text)

    # Reset state
    st.active = False
    st.agg = EmotionAgg()
    st.started_at = 0.0
    st.touch()

    return {"status": "success", "summary": {"report": report_text}}

# =========================
# Endpoints
# =========================
@app.get("/")
async def health():
    return {
        "status": "ok",
        "omi_creds_loaded": bool(OMI_APP_ID and OMI_APP_SECRET),
        "hume_ready": bool(HUME_API_KEY),
        "pid": PID
    }

@app.get("/hume/ping")
async def hume_ping():
    if not hume_client:
        return {"ok": False, "error": "Hume not configured"}
    # open/close a socket to validate connectivity (no media sent)
    try:
        options = StreamConnectOptions(config=EMConfig(prosody={}))
        async with hume_client.expression_measurement.stream.connect(options=options):
            pass
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/audio")
async def audio_ingest(
    request: Request,
    uid: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    sample_rate: int = Query(16000),
):
    """
    Omi posts raw audio bytes here every ~3–5s:
      POST /audio?sample_rate=16000&uid={{userId}}
    or (if you prefer session-based):
      POST /audio?sample_rate=16000&session_id={{sessionId}}

    Body: application/octet-stream (raw PCM16 mono bytes expected)
    """
    key = _effective_id(uid, session_id, request.headers)
    if not key:
        return {"status": "ignored", "reason": "missing_uid_or_session_id"}

    body = await request.body()
    if not body:
        return {"status": "ignored", "reason": "empty_body"}

    now = time.time()
    st = await _get_state(key)

    async with st.lock:
        # Idle finalize before new chunk if needed
        if st.active and st.last_wall_ts and (now - st.last_wall_ts > IDLE_TIMEOUT_SEC):
            print(f"⏰ Idle timeout for id={key} — auto-finalize")
            await finalize_and_report(st)

        if not st.active:
            st.active = True
            st.started_at = now
            st.agg = EmotionAgg()
            print(f"🟢 session starts (id={key})")

        st.touch()
        st.last_audio_ts = now

    # Estimate chunk duration assuming PCM16 mono
    approx_dur = len(body) / 2.0 / max(sample_rate, 1)

    # Send to Hume EM stream
    res = await hume_measure_bytes(body, sample_rate)
    if res:
        async with st.lock:
            st.agg.add_predictions(res, approx_dur, t0=now - approx_dur)
    else:
        print(f"⚠️ Hume returned no result for id={key}")

    return {"status": "ok", "id": key, "received": len(body), "sample_rate": sample_rate}

@app.post("/conversation/end")
async def conversation_end(uid: Optional[str] = Query(None), session_id: Optional[str] = Query(None)):
    key = uid or session_id
    if not key:
        return {"status": "ignored", "reason": "missing_uid_or_session_id"}
    st = await _get_state(key)
    async with st.lock:
        if not st.active:
            return {"status": "ignored", "reason": "not_active"}
        return await finalize_and_report(st)

# Entrypoint (local dev)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on http://0.0.0.0:{port} (pid={PID})")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
