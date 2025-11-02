# file: hume_vocal_tone_app.py
# --------------------------------------------------------------------
# Vocal Tone (Hume Prosody) -> LLM Report -> Push to Omi (notification, conversation, memory)
#
# Requires:
#   pip install "hume[stream]" fastapi uvicorn httpx python-dotenv pydantic
#
# ENV:
#   OMI_APP_ID         = <your Omi integration id>
#   OMI_APP_SECRET     = <your Omi app secret>     (for /notification)  <your Omi API key>        (for Imports: conversations/memories)
#   HUME_API_KEY       = <your Hume API key>
#   DEEPSEEK_API_KEY   = <your LLM key>            (DeepSeek or drop-in compatible chat API)
#   DEEPSEEK_MODEL     = deepseek-reasoner         (default)
#   DEEPSEEK_BASE_URL  = https://api.deepseek.com/v1
#
# Run:
#   uvicorn hume_vocal_tone_app:app --host 0.0.0.0 --port 10000
#   (Configure Omi to POST audio bytes to /audio?uid=...&sample_rate=16000 every ~3–5s)
# --------------------------------------------------------------------

import os
import re
import asyncio
import base64
import json
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Query
from pydantic import BaseModel, Field

# Hume SDK (WebSocket streaming)
from hume import HumeStreamClient
from hume.models.config import ProsodyConfig, BurstConfig

# =========================
# Env / constants
# =========================
OMI_APP_ID = os.environ.get("OMI_APP_ID")
OMI_APP_SECRET = os.environ.get("OMI_APP_SECRET")     # for notifications

HUME_API_KEY = os.environ.get("HUME_API_KEY")

# LLM (DeepSeek-style Chat Completions; drop-in if you swap to OpenAI-compatible)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# Session controls
IDLE_TIMEOUT_SEC = int(os.environ.get("IDLE_TIMEOUT_SEC", "180"))  # finalize if idle this long
MAX_CHUNK_SEC = float(os.environ.get("MAX_CHUNK_SEC", "4.8"))      # keep <5s for Hume stream slices
PID = os.getpid()

# Optional markers (not used by /audio, but handy if you later add text triggers)
START_RE = re.compile(r"\bconversation\s*starts\b", re.I)
END_RE   = re.compile(r"\bconversa(?:i?t)ion\s*end(?:s)?\b", re.I)

# =========================
# Models (if you add metadata)
# =========================
class RTIncomingSegment(BaseModel):
    id: Optional[str] = None
    text: Optional[str] = None
    speaker: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None

class RTTranscriptBatch(BaseModel):
    segments: List[RTIncomingSegment] = Field(default_factory=list)
    session_id: Optional[str] = None
    uid: Optional[str] = None

# =========================
# App & clients
# =========================
http_client: Optional[httpx.AsyncClient] = None
hume_client: Optional[HumeStreamClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, hume_client
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🚀 Hume Vocal Tone server starting...")
    http_client = httpx.AsyncClient(timeout=45.0)
    hume_client = HumeStreamClient(HUME_API_KEY) if HUME_API_KEY else None
    try:
        yield
    finally:
        if http_client:
            await http_client.aclose()
        print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 👋 Server shutting down...")

app = FastAPI(title="Vocal Tone (Hume Prosody x Omi x LLM)", lifespan=lifespan)

# =========================
# Omi helpers
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
            print(f"❌ LLM timeout (attempt {attempt+1})"); continue
        except Exception as e:
            print(f"❌ LLM network (attempt {attempt+1}): {e}")
            await asyncio.sleep(1.5 * (attempt + 1)); continue

        if resp.status_code // 100 != 2:
            txt = resp.text[:400]
            print(f"❌ LLM HTTP {resp.status_code} (attempt {attempt+1}): {txt}")
            if resp.status_code in (429, 500, 502, 503, 504):
                await asyncio.sleep(1.5 * (attempt + 1)); continue
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
   - Identify the the strongest emotions (by average or repeated high scores) for each speaker (if identifiable).
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

- Overall Positivity (0–100): higher when positive expressions (Joy, Amusement, Calmness, Contentment, Admiration, Relief, Tenderness, Interest) are frequent/strong.
- Vocal Confidence (0–100): higher with dynamic range and clear expressive peaks (not just sporadic spikes), lower for flat/strained patterns.
- Engagement Level (0–100): higher with strong/sustained Interest/Curiosity/Concentration; lower when signals are weak or inconsistent.
- Empathy (0–100): higher if multi-speaker mirroring/soothing patterns occur; otherwise estimate conservatively.

- Vocal Tone (0–100): a weighted blend (you decide a sensible weighting; positivity and confidence matter most).
- AI Confidence (0–100): base this on data sufficiency and quality:
  * longer total voiced duration and more consistent predictions → higher,
  * very short/sparse/noisy data → lower.

If data is sparse, STILL produce the full report with a low AI Confidence and cautious language.

---
REQUIRED OUTPUT FORMAT (VOCAL TONE MODE)
Return your analysis in EXACTLY this plain-text format:

Vocal Tone: [0–100] — “[Short, catchy title for the vibe]”
AI Confidence: [0–100]

[1–2 sentence human summary of the audible emotional dynamics]

✅ Highlights:
• [Strong point #1, e.g., "Consistent vocal warmth (Joy/Calmness) through the middle section."]
• [Strong point #2, e.g., "Noticeable Interest peaks during the other speaker’s turns."]

💡 Try:
• [Actionable tip #1, e.g., "Add more dynamic range; vary pitch and emphasis to avoid flat delivery."]
• [Actionable tip #2, e.g., "If tension rises (Frustration/Anxiety peaks), slow pacing and add small pauses."]

---
Breakdown:
• Overall Positivity: [0–100] ([very brief reason])
• Vocal Confidence: [0–100] ([very brief reason])
• Engagement Level: [0–100] ([very brief reason])
• Empathy: [0–100] ([very brief reason])
---
✨ Emotional Moments:
• [t_start–t_end or ~t_s]: "Speaker X showed high **[Emotion]**" — (1-line why this matters)
• [t_start–t_end or ~t_s]: "Speaker Y registered **[Emotion]**" — (1-line why this matters)

---
IMPORTANT
- Use only emotion names present in the JSON.
- If multiple speakers are not identifiable, do not fabricate them; phrase moments generically.
- Keep it concise, clear, and faithful to the data.
- Always output the full report, starting with 'Vocal Tone:'.
"""

def build_hume_user_prompt(hume_json: dict, title_hint: Optional[str] = None) -> str:
    """Inject your (possibly compressed) Hume JSON."""
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
        self.predictions: List[Dict[str, Any]] = []  # store raw-ish preds for LLM
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
            # Keep raw emotions for LLM
            emotions = p.get("emotions", [])
            # Track totals & peaks
            for e in emotions:
                name = str(e.get("name", "unknown")).strip()
                score = float(e.get("score", 0.0))
                self.emotion_sum[name] = self.emotion_sum.get(name, 0.0) + score
                if score >= 0.75:
                    self.peak_events.append((t0 + begin, name, score))
            # Store prediction
            self.predictions.append({
                "time": {"begin": begin + t0, "end": float(time_obj.get("end", begin)) + t0},
                "emotions": emotions,
                "speaker": p.get("speaker") or p.get("person") or None
            })
        self.total_dur += max(slice_dur, 1e-6)

    def compressed_json(self, topk_per_pred=5, min_score=0.15, max_preds=500) -> Dict[str, Any]:
        """
        Compress predictions for token efficiency:
        - keep top-K emotions per prediction
        - drop low scores
        - cap total number of predictions
        """
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
        return {
            "meta": {"audio_duration_sec": self.total_dur},
            "prosody": {"predictions": preds_out}
        }

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
# Hume streaming (per-chunk)
# =========================
async def hume_measure_bytes(audio_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Open a short-lived Hume stream, send a <=~5s slice, return the JSON result.
    """
    if not hume_client:
        print("❌ Hume client not configured")
        return None
    configs = [ProsodyConfig(), BurstConfig(use_boost=True)]
    async with hume_client.connect(configs) as socket:
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        result = await socket.send_bytes(b64)
        return result

# =========================
# Finalization: LLM report
# =========================
async def finalize_and_report(st: VoiceState, title_hint="Vocal Tone Session") -> Dict[str, Any]:
    uid = st.uid
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🟥 Finalizing uid={uid}")
    # Compress Hume JSON for LLM
    hume_json = st.agg.compressed_json(topk_per_pred=5, min_score=0.15, max_preds=600)
    # Build prompts
    sys_prompt = HUME_ANALYSIS_SYSTEM_PROMPT
    user_prompt = build_hume_user_prompt(hume_json, title_hint=title_hint)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    report_text = await call_llm(messages, temperature=0.2, max_tokens=1400)

    if not report_text:
        # Fallback: minimal deterministic report if LLM failed
        top = sorted(st.agg.emotion_sum.items(), key=lambda x: x[1], reverse=True)[:5]
        top_str = ", ".join([n for n, _ in top]) if top else "n/a"
        report_text = (
            f"Vocal Tone: 62 — \"Neutral to Warm\"\n"
            f"AI Confidence: {min(95, int(20 + 75*min(1.0, st.agg.total_dur/60.0)))}\n\n"
            f"Overview based on prosody only. Top expressions detected: {top_str}.\n\n"
            f"✅ Highlights:\n"
            f"• Clear vocal presence at times\n"
            f"• Some expressive peaks detected\n\n"
            f"💡 Try:\n"
            f"• Add more dynamic range to avoid flat delivery\n"
            f"• Slow down slightly if tension rises\n\n"
            f"Breakdown:\n"
            f"• Overall Positivity: 60 (mixed-positive cues)\n"
            f"• Vocal Confidence: 65 (moderately dynamic)\n"
            f"• Engagement Level: 60 (some interest cues)\n"
            f"• Empathy: 55 (uncertain without speaker separation)\n"
            f"---\n"
            f"✨ Emotional Moments:\n"
            f"• ~{int(st.agg.peak_events[0][0])}s: high **{st.agg.peak_events[0][1]}** (if available)\n"
            f"• ~{int(st.agg.peak_events[1][0])}s: high **{st.agg.peak_events[1][1]}** (if available)\n"
        )

    # Push back to Omi
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
        "pid": PID
    }

@app.post("/audio")
async def audio_ingest(request: Request, uid: str = Query(...), sample_rate: int = Query(16000)):
    """
    Omi posts raw audio bytes here: POST /audio?sample_rate=16000&uid=...  (octet-stream)
    In Omi Dev Settings, set "Every X seconds" to ~3–5s so each chunk is <5s.
    """
    st = await _get_state(uid)
    body = await request.body()
    now = time.time()

    # Guard: empty body
    if not body:
        return {"status": "ignored", "reason": "empty_body"}

    async with st.lock:
        # Idle finalize before new chunk (if needed)
        if st.active and st.last_wall_ts and (now - st.last_wall_ts > IDLE_TIMEOUT_SEC):
            print(f"⏰ Idle timeout for uid={uid} — auto-finalize")
            await finalize_and_report(st)

        if not st.active:
            st.active = True
            st.started_at = now
            st.agg = EmotionAgg()
            print(f"🟢 session starts (uid={uid})")

        st.touch()
        st.last_audio_ts = now

    # Send slice to Hume (short chunk)
    # Optional: estimate duration for weighting assuming PCM16 mono
    approx_dur = len(body) / 2.0 / max(sample_rate, 1)  # bytes / 2 / Hz
    res = await hume_measure_bytes(body)
    if res:
        async with st.lock:
            st.agg.add_predictions(res, approx_dur, t0=now - approx_dur)
    else:
        print(f"⚠️ Hume returned no result for uid={uid}")

    return {"status": "ok", "received": len(body)}

@app.post("/conversation/end")
async def conversation_end(uid: str = Query(...)):
    st = await _get_state(uid)
    async with st.lock:
        if not st.active:
            return {"status": "ignored", "reason": "not_active"}
        return await finalize_and_report(st)

# Optional: ping the LLM path
@app.get("/llm/ping")
async def llm_ping():
    msgs = [
        {"role": "system", "content": "Return ONLY: ok"},
        {"role": "user", "content": "ok"}
    ]
    out = await call_llm(msgs, temperature=0.0, max_tokens=5)
    return {"bridge_ok": out == "ok", "raw": out}

# Entrypoint
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Hume Vocal Tone server on http://0.0.0.0:{port} (pid={PID})")
    uvicorn.run("hume_vocal_tone_app:app", host="0.0.0.0", port=port, reload=False)
