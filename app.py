# --------------------------------------------------------------------
# Vocal Tone (Hume Prosody) -> LLM Report (DeepSeek) -> Push to Omi
# Deterministic fallback included (no LLM required if call fails)
#
# Requires:
#   pip install -r requirements.txt
#
# ENV:
#   OMI_APP_ID         = <your Omi integration id>
#   OMI_APP_SECRET     = <your Omi app secret>  (USED FOR ALL OMI ENDPOINTS)
#   HUME_API_KEY       = <your Hume API key>
#   DEEPSEEK_API_KEY   = <your LLM key>         (DeepSeek or OpenAI-compatible)
#   DEEPSEEK_MODEL     = deepseek-reasoner      (default)
#   DEEPSEEK_BASE_URL  = https://api.deepseek.com/v1
#
# Run (Render Start Command):
#   uvicorn app:app --host 0.0.0.0 --port $PORT
#   (In Omi Dev Settings, POST audio bytes to /audio?uid=...&sample_rate=16000)
# --------------------------------------------------------------------

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

# ---------------- Hume SDK (robust imports; no socket_client) ----------------
try:
    from hume.client import AsyncHumeClient  # preferred in current SDKs
except Exception:
    from hume import AsyncHumeClient  # fallback export if package root re-exports

try:
    from hume.expression_measurement.stream import Config as EMConfig
except Exception:
    EMConfig = None  # we'll fall back to a dict model config

try:
    # Some versions expose a generic data models builder
    from hume import StreamDataModels as HumeStreamDataModels  # type: ignore
except Exception:
    HumeStreamDataModels = None  # optional fallback

# ---------------------------------------------------------------------------

PID = os.getpid()

# =========================
# Env / constants
# =========================
OMI_APP_ID = os.environ.get("OMI_APP_ID")
OMI_APP_SECRET = os.environ.get("OMI_APP_SECRET")  # USED FOR ALL OMI CALLS

HUME_API_KEY = os.environ.get("HUME_API_KEY")

# LLM (DeepSeek-style Chat Completions; drop-in if you swap to OpenAI-compatible)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# Session controls
IDLE_TIMEOUT_SEC = int(os.environ.get("IDLE_TIMEOUT_SEC", "180"))  # finalize if idle this long
MAX_CHUNK_SEC = float(os.environ.get("MAX_CHUNK_SEC", "4.8"))      # keep <5s for Hume stream slices

# Optional markers (not used by /audio, but handy if you later add text triggers)
START_RE = re.compile(r"\bconversation\s*starts\b", re.I)
END_RE   = re.compile(r"\bconversa(?:i?t)ion\s*end(?:s)?\b", re.I)

# =========================
# App & clients
# =========================
http_client: Optional[httpx.AsyncClient] = None
hume_client: Optional[AsyncHumeClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, hume_client
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🚀 Hume Vocal Tone server starting...")
    http_client = httpx.AsyncClient(timeout=45.0)
    hume_client = AsyncHumeClient(api_key=HUME_API_KEY) if HUME_API_KEY else None
    try:
        yield
    finally:
        if http_client:
            await http_client.aclose()
        print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 👋 Server shutting down...")

app = FastAPI(title="Vocal Tone (Hume Prosody x Omi x DeepSeek)", lifespan=lifespan)

# =========================
# Omi helpers (use OMI_APP_SECRET for ALL endpoints)
# =========================
async def omi_create_conversation(uid: str, text: str):
    if not (OMI_APP_ID and OMI_APP_SECRET and http_client):
        return
    url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/user/conversations"
    headers = {"Authorization": f"Bearer {OMI_APP_SECRET}", "Content-Type": "application/json"}
    params = {"uid": uid}
    payload = {"text": text, "text_source": "other_text", "text_source_spec": "vocal_tone_report", "language": "en"}
    r = await http_client.post(url, headers=headers, params=params, json=payload)
    if r.status_code // 100 != 2:
        print(f"❌ Omi create_conversation {r.status_code} {r.text[:300]}")

async def omi_send_notification(uid: str, title: str, body: str):
    if not (OMI_APP_ID and OMI_APP_SECRET and http_client):
        return
    url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/notification"
    headers = {"Authorization": f"Bearer {OMI_APP_SECRET}", "Content-Type": "application/json", "Content-Length": "0"}
    # Keep push short-ish to avoid truncation on device
    params = {"uid": uid, "message": f"{title}: {body}"[:1800]}
    try:
        r = await http_client.post(url, headers=headers, params=params, timeout=15.0)
        if r.status_code // 100 != 2:
            print(f"❌ Omi notification {r.status_code} {r.text[:300]}")
    except Exception as e:
        print(f"❌ Omi notification error: {e}")

async def omi_save_memory(uid: str, full_text: str):
    if not (OMI_APP_ID and OMI_APP_SECRET and http_client):
        return
    try:
        # Ensure full text is stored as a conversation
        await omi_create_conversation(uid, full_text)

        # Create a short memory entry
        url = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/user/memories"
        headers = {"Authorization": f"Bearer {OMI_APP_SECRET}", "Content-Type": "application/json"}
        params = {"uid": uid}
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
    except Exception as e:
        print(f"❌ Omi memory err: {e}")

# =========================
# LLM (DeepSeek-style) helpers
# =========================
async def call_llm(messages: List[Dict[str, str]], temperature=0.2, max_tokens=1200) -> Optional[str]:
    """Chat Completions compatible call; returns raw text content or None."""
    if not (DEEPSEEK_API_KEY and http_client):
        print("❌ LLM: missing API key or HTTP client.")
        return None
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    url = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    for attempt in range(3):
        try:
            resp = await http_client.post(url, headers=headers, json=payload, timeout=200.0)
        except httpx.ReadTimeout:
            print(f"❌ LLM timeout (attempt {attempt+1})")
            continue
        except Exception as e:
            print(f"❌ LLM network (attempt {attempt+1}): {e}")
            await asyncio.sleep(1.5 * (attempt + 1))
            continue

        if resp.status_code // 100 != 2:
            txt = resp.text[:400]
            print(f"❌ LLM HTTP {resp.status_code} (attempt {attempt+1}): {txt}")
            if resp.status_code in (429, 500, 502, 503, 504):
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            return None

        try:
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content")
            if not content:
                print(f"❌ LLM returned empty content. Finish_reason={choice.get('finish_reason')}")
                return None
            return content.strip()
        except Exception as e:
            print(f"❌ LLM parse error: {e}. Preview: {resp.text[:300]}")
            return None
    print("❌ LLM: retries exhausted.")
    return None

# =========================
# Hume Analysis Prompt (voice-only)
# =========================
HUME_ANALYSIS_SYSTEM_PROMPT = """You are an expert vocal and emotional analyst.
Your job is to read a JSON object containing vocal prosody and emotion data from Hume AI
(“Expression Measurement / Prosody”) and return a single, formatted, plain-text "Vocal Tone Report".

CRITICAL RULES
- BASE YOUR ANALYSIS ONLY ON THE PROVIDED HUME JSON (prosody/emotions over time). DO NOT invent content or rely on words/transcripts.
- DO NOT output JSON. DO NOT add any commentary before or after the report.
- Your entire response MUST be the report itself, starting with "Vocal Tone:".

---
WHAT THE JSON LOOKS LIKE (robust assumptions)
- JSON may contain: prosody.predictions[*].time.{begin,end}, .emotions[*].{name,score}, and optional speaker/person identifiers.
- speaker/person identifiers might be missing; if so, treat it as a single speaker.
- Some slices may be silent or sparse; handle gracefully.
- Scores are continuous in [0,1]; higher = stronger expression.
- Time fields are in seconds.

---
YOUR GOAL & CONTEXT
Analyze the prosody/emotion signals to describe the conversation’s audible vibe. Focus on:

1) Dominant Emotions:
   - Identify the strongest emotions (by average or repeated high scores).
   - Use labels present in the JSON (e.g., Joy, Amusement, Interest, Calmness, Frustration, Irritation, Sadness, Anxiety, etc.).
   - Do NOT infer inner feelings; describe expressed, perceived vocal qualities only.

2) Emotional Flow:
   - Did positivity (e.g., Joy/Amusement/Calmness/Contentment/Admiration) increase or fade over time?
   - Note shifts toward negative tension (e.g., Frustration/Irritation/Distress/Anxiety/Sadness).

3) Vocal Confidence:
   - From prosodic expression patterns, judge if tone was dynamic/smooth vs. flat/strained.
   - Consider presence of expressive peaks and consistency over time.

4) Engagement:
   - Look for sustained “Interest/Curiosity/Concentration” signals.
   - Lower engagement could manifest as consistently low or flat affect with no peaks.

5) Empathy (if multi-speaker):
   - Check for rough mirroring: if one shows Sadness/Distress, did the other show Tenderness/Concern/Calmness nearby in time?
   - If you cannot tell speakers apart, say so and rate Empathy conservatively.

---
HOW TO COMPUTE THE SCORES
Compute 0–100 sub-scores and a 0–100 aggregate "Vocal Tone" score. Be consistent:

- Overall Positivity (0–100): higher when positive expressions are frequent/strong.
- Vocal Confidence (0–100): higher with dynamic range and clear expressive peaks, lower for flat/strained patterns.
- Engagement Level (0–100): higher with strong/sustained Interest signals.
- Empathy (0–100): higher with mirroring/soothing patterns; otherwise estimate conservatively.

- Vocal Tone (0–100): a weighted blend (positivity & confidence matter most).
- AI Confidence (0–100): base this on data sufficiency/quality (more voiced duration → higher).

If data is sparse, STILL produce the full report with a low AI Confidence and cautious language.

---
REQUIRED OUTPUT FORMAT (VOCAL TONE MODE)
Return your analysis in EXACTLY this plain-text format:

Vocal Tone: [0–100] — “[Short, catchy title for the vibe]”
AI Confidence: [0–100]

[1–2 sentence human summary of the audible emotional dynamics]

✅ Highlights:
• [Strong point #1]
• [Strong point #2]

💡 Try:
• [Actionable tip #1]
• [Actionable tip #2]

---
Breakdown:
• Overall Positivity: [0–100] ([brief reason])
• Vocal Confidence: [0–100] ([brief reason])
• Engagement Level: [0–100] ([brief reason])
• Empathy: [0–100] ([brief reason])
---
✨ Emotional Moments:
• [~t_s or t_start–t_end]: high **[Emotion]** — (why it matters)
• [~t_s or t_start–t_end]: **[Emotion]** — (why it matters)

---
IMPORTANT
- Use only emotion names present in the JSON.
- If speakers aren’t identifiable, write moments generically.
- Keep it concise, clear, and faithful to the data.
- Always start with 'Vocal Tone:'.
"""

def build_hume_user_prompt(hume_json: dict, title_hint: Optional[str] = None) -> str:
    safe_hint = f"Title hint (optional): {title_hint}" if title_hint else "Title hint (optional):"
    body = json.dumps(hume_json, ensure_ascii=False)
    return f"""{safe_hint}

Hume Prosody JSON:
{body}
"""

# =========================
# Aggregation & compression
# =========================
class EmotionAgg:
    """
    Rolling aggregator of Hume prosody outputs across slices.
    Also stores full predictions for the LLM (with light compression later).
    """
    def __init__(self):
        self.total_dur = 0.0
        self.predictions: List[Dict[str, Any]] = []  # store raw predictions for LLM
        self.emotion_sum: Dict[str, float] = {}
        self.peak_events: List[Tuple[float, str, float]] = []  # (timestamp, name, score)

    def add_predictions(self, result: Dict[str, Any], slice_dur: float, t0: float):
        prosody = result.get("prosody") or result.get("models", {}).get("prosody")
        if not prosody:
            return
        preds = prosody.get("predictions", [])
        for p in preds:
            time_obj = p.get("time", {})
            begin = float(time_obj.get("begin", 0.0))
            end_v = float(time_obj.get("end", begin))
            emotions = p.get("emotions", [])
            # track totals & peaks
            for e in emotions:
                name = str(e.get("name", "unknown")).strip()
                score = float(e.get("score", 0.0))
                self.emotion_sum[name] = self.emotion_sum.get(name, 0.0) + score
                if score >= 0.75:
                    self.peak_events.append((t0 + begin, name, score))
            # keep raw-ish for LLM
            self.predictions.append({
                "time": {"begin": begin + t0, "end": end_v + t0},
                "emotions": [{"name": e.get("name"), "score": float(e.get("score", 0.0))} for e in emotions],
                "speaker": p.get("speaker") or p.get("person") or None
            })
        self.total_dur += max(slice_dur, 1e-6)

    def compressed_json(self, topk_per_pred=5, min_score=0.15, max_preds=600) -> Dict[str, Any]:
        preds_out = []
        for p in self.predictions[:max_preds]:
            emos = p.get("emotions", [])
            emos = [e for e in emos if float(e.get("score", 0.0)) >= min_score]
            emos = sorted(emos, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:topk_per_pred]
            preds_out.append({
                "speaker": p.get("speaker"),
                "time": p.get("time"),
                "emotions": [{"name": e.get("name"), "score": float(e.get("score", 0.0))} for e in emos]
            })
        return {"meta": {"audio_duration_sec": self.total_dur}, "prosody": {"predictions": preds_out}}

# Deterministic scoring (fallback)
POSITIVE = {"Admiration","Amusement","Awe","Calmness","Contentment","Interest","Joy","Relief","Triumph","Tenderness"}
TENSION  = {"Annoyance","Anxiety","Distress","Frustration","Irritation","Sadness","Disappointment","Contempt","Anger","Doubt"}
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
    empathy  = 50
    if agg.peak_events:
        has_tension  = any(n in TENSION for _, n, _ in agg.peak_events)
        has_positive = any(n in POSITIVE for _, n, _ in agg.peak_events)
        empathy = 70 if (has_tension and has_positive) else (60 if has_positive else 50)
    vocal_conf = max(0, min(100, int(0.6*energy + 0.4*(100 - tension))))
    vocal_tone = max(0, min(100, int(0.5*warmth + 0.3*vocal_conf + 0.2*engage - 0.1*tension)))
    ai_conf    = min(95, int(20 + 75 * min(1.0, agg.total_dur / 60.0)))
    return {"warmth": warmth, "tension": tension, "energy": energy, "engage": engage,
            "vocal_conf": vocal_conf, "vocal_tone": vocal_tone, "ai_conf": ai_conf}

def build_report(agg: EmotionAgg, title_hint: str = "Vocal Tone Session") -> str:
    m = compute_metrics(agg)
    top = sorted(agg.emotion_sum.items(), key=lambda x: x[1], reverse=True)[:2]
    top_lines = [f"• Strong **{n}** signal" for n, _ in top] or ["• Clear articulation", "• Stable prosody"]
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
        f'• Overall Positivity: {m["warmth"]}\n'
        f'• Vocal Confidence: {m["vocal_conf"]}\n'
        f'• Engagement Level: {m["engage"]}\n'
        f'• Empathy: {max(0,min(100,int((m["warmth"]+100-m["tension"])/2)))}\n'
        f'---\n'
        f'✨ Emotional Moments:\n'
        f'{(peak_lines[0] if peak_lines else "• Moments were evenly expressed")}\n'
        f'{(peak_lines[1] if len(peak_lines)>1 else "")}\n'
        f'\n'
        f'*Report derived from vocal expressions only; labels reflect perceived vocal tone, not inner feelings.*'
    )

# =========================
# Per-UID session state
# =========================
class VoiceState:
    def __init__(self, uid: str):
        self.uid = uid
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

async def _get_state(uid: str) -> VoiceState:
    async with STATES_LOCK:
        if uid not in STATES:
            STATES[uid] = VoiceState(uid)
        return STATES[uid]

# =========================
# Hume streaming utilities
# =========================
def _bytes_to_wav_path(raw: bytes, sample_rate: int) -> str:
    """Wrap raw PCM16 mono bytes with a WAV header to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # PCM16
        wf.setframerate(sample_rate)
        wf.writeframes(raw)
    return tmp_path

def _build_stream_config() -> Any:
    """Build Hume stream 'config' object in a version-resilient way."""
    if EMConfig is not None:
        return EMConfig(prosody={})  # Return the object directly
    if HumeStreamDataModels is not None:
        return HumeStreamDataModels(prosody={})  # Return the object directly
    return {"prosody": {}}  # Fallback to a plain dict

async def hume_measure_bytes(audio_bytes: bytes, sample_rate: int) -> Optional[Dict[str, Any]]:
    """Use Hume Expression Measurement (Prosody) stream for a short chunk."""
    if not hume_client:
        print("❌ Hume client not configured")
        return None
    
    # 1. Get the config object from the fixed helper
    stream_config = _build_stream_config() 
    
    wav_path = _bytes_to_wav_path(audio_bytes, sample_rate)
    try:
        # 2. Call connect() with NO arguments
        async with hume_client.expression_measurement.stream.connect() as socket:
            # 3. Pass the config to send_file() instead
            result = await socket.send_file(wav_path, config=stream_config)
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
# Finalization: Build report (LLM first, deterministic fallback)
# =========================
async def finalize_and_report(st: VoiceState, title_hint="Vocal Tone Session") -> Dict[str, Any]:
    uid = st.uid
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🟥 Finalizing uid={uid}")

    # 1) Try LLM-based report
    report_text: Optional[str] = None
    try:
        hume_json = st.agg.compressed_json(topk_per_pred=5, min_score=0.15, max_preds=600)
        sys_prompt = HUME_ANALYSIS_SYSTEM_PROMPT
        user_prompt = build_hume_user_prompt(hume_json, title_hint=title_hint)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        report_text = await call_llm(messages, temperature=0.2, max_tokens=1400)
    except Exception as e:
        print(f"❌ LLM pipeline error: {e}")

    # 2) Fallback deterministic report if LLM missing/fails
    if not report_text:
        report_text = build_report(st.agg, title_hint=title_hint)

    # Push to Omi (ALL endpoints use OMI_APP_SECRET)
    await omi_send_notification(uid, title="Your Vocal Tone Report", body=report_text[:800])
    await omi_create_conversation(uid, report_text)
    await omi_save_memory(uid, report_text)

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
        "llm_ready": bool(DEEPSEEK_API_KEY),
        "pid": PID,
    }

@app.get("/hume/ping")
async def hume_ping():
    if not hume_client:
        return {"ok": False, "error": "Hume not configured"}
    try:
        # Call connect() with NO arguments
        async with hume_client.expression_measurement.stream.connect():
            pass
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
        
@app.get("/llm/ping")
async def llm_ping():
    msgs = [{"role": "system", "content": "Return ONLY: ok"}, {"role": "user", "content": "ok"}]
    out = await call_llm(msgs, temperature=0.0, max_tokens=5)
    return {"bridge_ok": out == "ok", "raw": out}

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
    or
      POST /audio?sample_rate=16000&session_id={{sessionId}}
    Body: application/octet-stream (raw PCM16 mono bytes)
    """
    key = uid or session_id or request.headers.get("X-Session-ID") or request.headers.get("X-Omi-Uid")
    if not key:
        return {"status": "ignored", "reason": "missing_uid_or_session_id"}

    st = await _get_state(key)
    body = await request.body()
    now = time.time()

    if not body:
        return {"status": "ignored", "reason": "empty_body"}

    async with st.lock:
        # Idle finalize before new chunk (if needed)
        if st.active and st.last_wall_ts and (now - st.last_wall_ts > IDLE_TIMEOUT_SEC):
            print(f"⏰ Idle timeout for uid={key} — auto-finalize")
            await finalize_and_report(st)

        if not st.active:
            st.active = True
            st.started_at = now
            st.agg = EmotionAgg()
            print(f"🟢 session starts (uid={key})")

        st.touch()
        st.last_audio_ts = now

    # Estimate duration assuming PCM16 mono
    approx_dur = len(body) / 2.0 / max(sample_rate, 1)

    # Send slice to Hume
    res = await hume_measure_bytes(body, sample_rate)
    if res:
        async with st.lock:
            st.agg.add_predictions(res, approx_dur, t0=now - approx_dur)
    else:
        print(f"⚠️ Hume returned no result for uid={key}")

    return {"status": "ok", "received": len(body), "sample_rate": sample_rate}

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

# Entrypoint
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Hume Vocal Tone server on http://0.0.0.0:{port} (pid={PID})")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
