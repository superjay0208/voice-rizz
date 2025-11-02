# --------------------------------------------------------------------
# Vocal Tone (Hume Batch) -> LLM Report (DeepSeek) -> Push to Omi
# This version buffers all audio and uses the BATCH API at the end
# of the conversation.
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

# ---------------- Hume SDK (robust imports) ----------------
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

# Import for Error Handling (though we'll use batch, it's good practice)
try:
    from hume.models.stream import StreamErrorMessage
except ImportError:
    try:
        from hume.expression_measurement.stream.stream_socket import StreamErrorMessage
    except ImportError:
        StreamErrorMessage = None
# -----------------------------------------------------------

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
IDLE_TIMEOUT_SEC = int(os.environ.get("IDLE_TIMEOUT_SEC", "30"))  # finalize if idle this long

# NOTE: MAX_CHUNK_SEC is no longer relevant as we are not streaming to Hume
# MAX_CHUNK_SEC = float(os.environ.get("MAX_CHUNK_SEC", "4.8"))

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
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🚀 Hume Vocal Tone server (BATCH MODE) starting...")
    http_client = httpx.AsyncClient(timeout=45.0)
    hume_client = AsyncHumeClient(api_key=HUME_API_KEY) if HUME_API_KEY else None
    try:
        yield
    finally:
        if http_client:
            await http_client.aclose()
        print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 👋 Server shutting down...")

app = FastAPI(title="Vocal Tone (Hume Batch x Omi x DeepSeek)", lifespan=lifespan)

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
# NEW: Batch Result Processor
# (This replaces the old streaming 'EmotionAgg')
# =========================
class BatchResultProcessor:
    """
    Parses a full Hume BATCH API response.
    Populates attributes needed for both deterministic and LLM reports.
    Designed as a drop-in replacement for the old 'EmotionAgg' class.
    """
    def __init__(self, batch_result_json: List[Dict[str, Any]]):
        self.total_dur = 0.0
        self.predictions: List[Dict[str, Any]] = []  # For LLM
        self.emotion_sum: Dict[str, float] = {}      # For deterministic fallback
        self.peak_events: List[Tuple[float, str, float]] = []  # For deterministic fallback
        
        try:
            self._parse_batch_json(batch_result_json)
        except Exception as e:
            print(f"❌ BatchResultProcessor: Failed to parse Hume batch JSON: {e}")
            print(f"Data dump (first 1000 chars): {json.dumps(batch_result_json, indent=2)[:1000]}")

    def _parse_batch_json(self, batch_result_json: List[Dict[str, Any]]):
        # Batch API returns a list, one item per source file. We only sent one.
        if not batch_result_json:
            print("❌ BatchResultProcessor: JSON is empty.")
            return

        # Get the prosody predictions for the first file
        file_result = batch_result_json[0]
        if 'results' not in file_result or 'predictions' not in file_result['results']:
            print(f"❌ BatchResultProcessor: 'results' or 'predictions' not in {file_result.keys()}")
            return
            
        first_prediction_set = file_result['results']['predictions'][0]
        if 'models' not in first_prediction_set or 'prosody' not in first_prediction_set['models']:
            print(f"❌ BatchResultProcessor: 'models' or 'prosody' not in {first_prediction_set.keys()}")
            return
            
        prosody_data = first_prediction_set['models']['prosody']
        # Prefer 'predictions' if present; otherwise flatten 'grouped_predictions'
        all_preds = prosody_data.get('predictions')
        if not all_preds:
            gps = prosody_data.get('grouped_predictions', [])
            flat = []
            for g in gps:
                plist = g.get('predictions', [])
                # carry group id (speaker diarization sometimes goes here)
                gid = g.get('id') or g.get('speaker') or None
                for p in plist:
                    # normalize to the same shape we use elsewhere
                    p = dict(p)
                    p['speaker'] = p.get('speaker') or gid
                    flat.append(p)
            all_preds = flat

        if not all_preds:
            print("BatchResultProcessor: Prosody model ran but found no predictions.")
            return


        # Find total duration
        self.total_dur = max(float(p.get('time', {}).get('end', 0.0)) for p in all_preds)

        for p in all_preds:
            time_obj = p.get("time", {})
            begin = float(time_obj.get("begin", 0.0))
            end_v = float(time_obj.get("end", begin))
            
            emotions = p.get("emotions", [])
            
            # 1. Populate data for deterministic fallback
            for e in emotions:
                name = str(e.get("name", "unknown")).strip()
                score = float(e.get("score", 0.0))
                # Note: This is an unweighted sum, matching the old streaming logic.
                self.emotion_sum[name] = self.emotion_sum.get(name, 0.0) + score
                if score >= 0.75:
                    self.peak_events.append((begin, name, score)) # Use relative 'begin' timestamp
            
            # 2. Populate data for LLM analysis (this is the same structure as streaming)
            p_speaker = p.get("speaker") or p.get("person") or None
            p_emotions_list = []
            for e in emotions:
                p_emotions_list.append({
                    "name": e.get("name", "unknown"),
                    "score": float(e.get("score", 0.0))
                })

            self.predictions.append({
                "time": {"begin": begin, "end": end_v},
                "emotions": p_emotions_list,
                "speaker": p_speaker
            })

    def compressed_json(self, topk_per_pred=5, min_score=0.15, max_preds=600) -> Dict[str, Any]:
        """Compresses the full prediction list for the LLM prompt."""
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

# =========================
# Deterministic scoring (fallback)
# (This section is unchanged, it now reads from BatchResultProcessor)
# =========================
POSITIVE = {"Admiration","Amusement","Awe","Calmness","Contentment","Interest","Joy","Relief","Triumph","Tenderness"}
TENSION  = {"Annoyance","Anxiety","Distress","Frustration","Irritation","Sadness","Disappointment","Contempt","Anger","Doubt"}
ENERGY   = {"Excitement","Elation","Determination","Surprise"}
ENGAGE   = {"Interest","Curiosity","Concentration","Enthusiasm","Attentiveness"}

def _avg_over(agg: BatchResultProcessor, names: set) -> float:
    if agg.total_dur <= 0:
        return 0.0
    # Note: This was always (sum of scores) / (total duration).
    # This is not a true average, but it's consistent with the old code.
    s = sum(agg.emotion_sum.get(n, 0.0) for n in names)
    return 100.0 * (s / max(agg.total_dur, 1e-6))

def compute_metrics(agg: BatchResultProcessor) -> dict:
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

def build_report(agg: BatchResultProcessor, title_hint: str = "Vocal Tone Session") -> str:
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
# (MODIFIED FOR BATCH)
# =========================
class VoiceState:
    def __init__(self, uid: str):
        self.uid = uid
        self.active = False
        self.started_at = 0.0
        self.last_wall_ts = 0.0
        self.last_audio_ts = 0.0
        self.audio_buffer: List[bytes] = [] # MOD: This replaces EmotionAgg
        self.sample_rate: int = 16000       # MOD: Store sample rate
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
# Hume BATCH utilities
# (REPLACES streaming utilities)
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
    """
    Build Hume stream 'config' object in a version-resilient way.
    This is also used by the batch API's `submit_file` config.
    """
    if EMConfig is not None:
        return EMConfig(prosody={})  # Return the object directly
    if HumeStreamDataModels is not None:
        return HumeStreamDataModels(prosody={})  # Return the object directly
    return {"prosody": {}}  # Fallback to a plain dict

async def hume_measure_batch(audio_bytes: bytes, sample_rate: int) -> Optional[List[Dict[str, Any]]]:
    """
    Call Hume Expression Measurement (Batch) via REST.
    Returns the JSON predictions list (same shape you expected), or None on failure.
    """
    if not (HUME_API_KEY and http_client):
        print("❌ Hume REST: missing API key or HTTP client")
        return None

    # 1) Write a WAV temp from raw PCM16 (mono)
    wav_path = _bytes_to_wav_path(audio_bytes, sample_rate)

    headers = {
        "X-Hume-Api-Key": HUME_API_KEY,
        # NOTE: httpx sets multipart boundary/Content-Type automatically when using files=...
    }
    # Prosody-only job
    models = {"models": {"prosody": {}}}

    # 2) Start job (multipart: json + file)
    job_id = None
    f = None
    try:
        f = open(wav_path, "rb")
        files = {
            "json": (None, json.dumps(models), "application/json"),
            "file": (os.path.basename(wav_path), f, "audio/wav"),
        }
        resp = await http_client.post(
            "https://api.hume.ai/v0/batch/jobs",
            headers=headers,
            files=files,
            timeout=60.0,
        )
        if resp.status_code // 100 != 2:
            print(f"❌ Hume REST: job submit failed {resp.status_code} {resp.text[:400]}")
            return None
        job_id = resp.json().get("job_id")
        if not job_id:
            print(f"❌ Hume REST: job_id missing in response: {resp.text[:300]}")
            return None
    except Exception as e:
        print(f"❌ Hume REST: submit error: {e}")
        return None
    finally:
        try:
            if f: f.close()
            os.remove(wav_path)
        except Exception:
            pass

    # 3) Poll job status with backoff (recommend webhooks for prod)
    status = "PENDING"
    started = time.time()
    delay = 1.0
    while True:
        try:
            detail = await http_client.get(
                f"https://api.hume.ai/v0/batch/jobs/{job_id}",
                headers=headers,
                timeout=20.0,
            )
            if detail.status_code // 100 != 2:
                print(f"⚠️ Hume REST: status {detail.status_code} {detail.text[:200]}")
            else:
                j = detail.json()
                status = (j.get("state") or {}).get("status", status)
                if status == "COMPLETED":
                    break
                if status == "FAILED":
                    print(f"❌ Hume REST: job {job_id} failed.")
                    return None
        except Exception as e:
            print(f"⚠️ Hume REST: poll error {e}")

        if time.time() - started > 600:  # 10 min hard timeout
            print(f"❌ Hume REST: job {job_id} timed out.")
            return None
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, 6.0)

    # 4) Fetch predictions JSON
    try:
        preds = await http_client.get(
            f"https://api.hume.ai/v0/batch/jobs/{job_id}/predictions",
            headers=headers,
            timeout=60.0,
        )
        if preds.status_code // 100 != 2:
            print(f"❌ Hume REST: predictions fetch failed {preds.status_code} {preds.text[:400]}")
            return None
        return preds.json()
    except Exception as e:
        print(f"❌ Hume REST: predictions error {e}")
        return None

# =========================
# Finalization: Build report
# (MODIFIED FOR BATCH)
# =========================
async def finalize_and_report(st: VoiceState, title_hint="Vocal Tone Session") -> Dict[str, Any]:
    uid = st.uid
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🟥 Finalizing uid={uid}")

    # --- BATCH LOGIC START ---
    if not st.audio_buffer:
        print(f"⚠️ No audio buffered for uid={uid}. Skipping report.")
        # Reset state anyway
        st.active = False
        st.started_at = 0.0
        st.touch()
        return {"status": "ignored", "reason": "no_audio_buffered"}

    print(f"Combining {len(st.audio_buffer)} audio chunks for batch processing...")
    full_audio_bytes = b"".join(st.audio_buffer)
    sample_rate = st.sample_rate # Get sample rate from state
    
    # Reset buffer *before* long-running API call
    st.audio_buffer = [] 
    
    # Call Batch API (this can take time)
    batch_result_json = await hume_measure_batch(full_audio_bytes, sample_rate)

    if not batch_result_json:
        print(f"❌ Hume Batch job failed or returned empty for uid={uid}.")
        # Fallback to a very simple deterministic report
        report_text = f'Vocal Tone: -- — "{title_hint}"\nAI Confidence: 0\n\nFailed to process audio with Hume Batch API.'
    else:
        # NEW: Process the full batch JSON using our new parser class
        processor = BatchResultProcessor(batch_result_json)
        
        # 1) Try LLM-based report
        report_text: Optional[str] = None
        try:
            hume_json_for_llm = processor.compressed_json(topk_per_pred=5, min_score=0.15, max_preds=600)
            sys_prompt = HUME_ANALYSIS_SYSTEM_PROMPT
            user_prompt = build_hume_user_prompt(hume_json_for_llm, title_hint=title_hint)
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
            report_text = await call_llm(messages, temperature=0.2, max_tokens=1400)
        except Exception as e:
            print(f"❌ LLM pipeline error: {e}")

        # 2) Fallback deterministic report if LLM missing/fails
        if not report_text:
            report_text = build_report(processor, title_hint=title_hint) # Pass the processor object
    # --- BATCH LOGIC END ---

    # Push to Omi (ALL endpoints use OMI_APP_SECRET)
    await omi_send_notification(uid, title="Your Vocal Tone Report", body=report_text[:800])
    await omi_create_conversation(uid, report_text)
    await omi_save_memory(uid, report_text)

    # Reset state
    st.active = False
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
        "mode": "batch",
    }

@app.get("/hume/ping")
async def hume_ping():
    """Pings the BATCH client by listing recent jobs."""
    if not hume_client:
        return {"ok": False, "error": "Hume not configured"}
    try:
        # Pinging batch is safer; just list jobs
        await hume_client.expression_measurement.batch.list_jobs(limit=1)
        return {"ok": True, "ping_target": "batch_list_jobs"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/llm/ping")
async def llm_ping():
    msgs = [{"role": "system", "content": "Return ONLY: ok"}, {"role": "user", "content": "ok"}]
    out = await call_llm(msgs, temperature=0.0, max_tokens=5)
    return {"bridge_ok": out == "ok", "raw": out}

# --- VoiceState ---
class VoiceState:
    def __init__(self, uid: str):
        ...
        self.codec: str = "pcm16"   # ← new

# --- /audio endpoint ---
@app.post("/audio")
async def audio_ingest(
    request: Request,
    uid: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    sample_rate: int = Query(16000),
    codec: str = Query("pcm16"),  # ← new: 'pcm16' | 'pcm8' | 'opus' | 'opusfs320'
):
    ...
    async with st.lock:
        if not st.active:
            ...
            st.sample_rate = sample_rate
            st.codec = codec.lower()  # ← new
            print(f"🟢 session starts (uid={key}, sample_rate={sample_rate}, codec={st.codec})")
        ...
        st.audio_buffer.append(body)
    return {"status": "ok_buffered", "received": len(body), "sample_rate": sample_rate, "codec": codec}

# --- helper: convert PCM8 → PCM16 (μ-law/ALAW not handled; this is linear 8-bit) ---
def _pcm8_to_pcm16(pcm8: bytes) -> bytes:
    import array
    a = array.array('B', pcm8)                   # unsigned 8-bit
    # convert to signed 16-bit: center at 0 and scale
    out = array.array('h', ((x - 128) << 8 for x in a))
    return out.tobytes()

# --- in finalize_and_report before _bytes_to_wav_path(...) ---
full_audio_bytes = b"".join(st.audio_buffer)
if st.codec == "pcm8":                          # ← new
    full_audio_bytes = _pcm8_to_pcm16(full_audio_bytes)
elif st.codec in ("opus", "opusfs320"):         # ← new
    # Easiest: configure Omi sender to PCM16. Otherwise, decode here.
    print("❌ Received Opus but decoder not enabled; please send PCM16 or add Opus decode.")
    # return an error or proceed after adding an Opus decoder.

@app.post("/conversation/end")
async def conversation_end(uid: Optional[str] = Query(None), session_id: Optional[str] = Query(None)):
    key = uid or session_id
    if not key:
        return {"status": "ignored", "reason": "missing_uid_or_session_id"}
    st = await _get_state(key)
    async with st.lock:
        if not st.active:
            return {"status": "ignored", "reason": "not_active"}
        # This will now trigger the full batch process
        return await finalize_and_report(st)

# Entrypoint
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Hume Vocal Tone server (BATCH MODE) on http://0.0.0.0:{port} (pid={PID})")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
