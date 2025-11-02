# --------------------------------------------------------------------
# Vocal Tone (Hume Batch) -> LLM Report (DeepSeek) -> Push to Omi
# Buffers audio and uses Hume BATCH REST at the end of a session.
#
# Run (Render Start Command):
#   uvicorn app:app --host 0.0.0.0 --port $PORT
# Omi sender:
#   POST raw audio bytes to /audio?uid=...&sample_rate=16000&codec=pcm16
# --------------------------------------------------------------------

import os
import re
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

# ---------------- Optional Hume SDK (for /hume/ping only) ----------------
try:
    from hume.client import AsyncHumeClient  # recent SDKs
except Exception:
    try:
        from hume import AsyncHumeClient      # some versions re-export
    except Exception:
        AsyncHumeClient = None  # type: ignore
try:
    from hume.expression_measurement.stream import Config as EMConfig
except Exception:
    EMConfig = None
try:
    from hume import StreamDataModels as HumeStreamDataModels  # type: ignore
except Exception:
    HumeStreamDataModels = None
# -------------------------------------------------------------------------

PID = os.getpid()

# =========================
# Env / constants
# =========================
OMI_APP_ID = os.environ.get("OMI_APP_ID")
OMI_APP_SECRET = os.environ.get("OMI_APP_SECRET")  # USED FOR ALL OMI ENDPOINTS

HUME_API_KEY = os.environ.get("HUME_API_KEY")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

IDLE_TIMEOUT_SEC = int(os.environ.get("IDLE_TIMEOUT_SEC", "30"))

START_RE = re.compile(r"\bconversation\s*starts\b", re.I)
END_RE   = re.compile(r"\bconversa(?:i?t)ion\s*end(?:s)?\b", re.I)

# =========================
# App & clients
# =========================
http_client: Optional[httpx.AsyncClient] = None
hume_client: Optional[Any] = None  # only for /hume/ping

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, hume_client
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🚀 Hume Vocal Tone server (BATCH MODE) starting...")
    http_client = httpx.AsyncClient(timeout=45.0)
    if AsyncHumeClient and HUME_API_KEY:
        try:
            hume_client = AsyncHumeClient(api_key=HUME_API_KEY)
        except Exception:
            hume_client = None
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
        await omi_create_conversation(uid, full_text)
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
    if not (DEEPSEEK_API_KEY and http_client):
        print("❌ LLM: missing API key or HTTP client.")
        return None
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": DEEPSEEK_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
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
[...trimmed for brevity — keep your full prompt here...]
"""

def build_hume_user_prompt(hume_json: dict, title_hint: Optional[str] = None) -> str:
    safe_hint = f"Title hint (optional): {title_hint}" if title_hint else "Title hint (optional):"
    body = json.dumps(hume_json, ensure_ascii=False)
    return f"""{safe_hint}

Hume Prosody JSON:
{body}
"""

# =========================
# Batch Result Processor
# =========================
class BatchResultProcessor:
    def __init__(self, batch_result_json: List[Dict[str, Any]]):
        self.total_dur = 0.0
        self.predictions: List[Dict[str, Any]] = []
        self.emotion_sum: Dict[str, float] = {}
        self.peak_events: List[Tuple[float, str, float]] = []
        try:
            self._parse_batch_json(batch_result_json)
        except Exception as e:
            print(f"❌ BatchResultProcessor parse error: {e}")
            print(f"Data preview: {json.dumps(batch_result_json, indent=2)[:1000]}")

    def _parse_batch_json(self, batch_result_json: List[Dict[str, Any]]):
        if not batch_result_json:
            print("❌ BatchResultProcessor: JSON empty."); return
        file_result = batch_result_json[0]
        res = file_result.get("results", {})
        pred_sets = (res.get("predictions") or [])
        if not pred_sets:
            print("❌ BatchResultProcessor: results.predictions missing/empty."); return
        first_prediction_set = pred_sets[0]
        models = first_prediction_set.get("models", {})
        if "prosody" not in models:
            print("❌ BatchResultProcessor: models.prosody missing."); return

        prosody = models["prosody"]

        # Prefer flat predictions, else flatten grouped_predictions
        all_preds = prosody.get("predictions")
        if not all_preds:
            flat: List[Dict[str, Any]] = []
            for g in prosody.get("grouped_predictions", []):
                gid = g.get("id") or g.get("speaker")
                for p in g.get("predictions", []):
                    q = dict(p)
                    q["speaker"] = q.get("speaker") or gid
                    flat.append(q)
            all_preds = flat

        if not all_preds:
            print("BatchResultProcessor: Prosody model ran but found no predictions."); return

        self.total_dur = max(float(p.get("time", {}).get("end", 0.0)) for p in all_preds if p.get("time"))

        for p in all_preds:
            t = p.get("time", {}) or {}
            begin = float(t.get("begin", 0.0)); end_v = float(t.get("end", begin))
            emotions = p.get("emotions", []) or []
            for e in emotions:
                name = str(e.get("name", "unknown")).strip()
                score = float(e.get("score", 0.0))
                self.emotion_sum[name] = self.emotion_sum.get(name, 0.0) + score
                if score >= 0.75:
                    self.peak_events.append((begin, name, score))
            self.predictions.append({
                "time": {"begin": begin, "end": end_v},
                "emotions": [{"name": e.get("name"), "score": float(e.get("score", 0.0))} for e in emotions],
                "speaker": p.get("speaker") or p.get("person") or None
            })

    def compressed_json(self, topk_per_pred=5, min_score=0.15, max_preds=600) -> Dict[str, Any]:
        preds_out = []
        for p in self.predictions[:max_preds]:
            emos = [e for e in p.get("emotions", []) if float(e.get("score", 0.0)) >= min_score]
            emos = sorted(emos, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:topk_per_pred]
            preds_out.append({"speaker": p.get("speaker"), "time": p.get("time"), "emotions": emos})
        return {"meta": {"audio_duration_sec": self.total_dur}, "prosody": {"predictions": preds_out}}

# =========================
# Deterministic scoring (fallback)
# =========================
POSITIVE = {"Admiration","Amusement","Awe","Calmness","Contentment","Interest","Joy","Relief","Triumph","Tenderness"}
TENSION  = {"Annoyance","Anxiety","Distress","Frustration","Irritation","Sadness","Disappointment","Contempt","Anger","Doubt"}
ENERGY   = {"Excitement","Elation","Determination","Surprise"}
ENGAGE   = {"Interest","Curiosity","Concentration","Enthusiasm","Attentiveness"}

def _avg_over(agg: BatchResultProcessor, names: set) -> float:
    if agg.total_dur <= 0: return 0.0
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
    peak_lines = [f'• ~{int(ts)}s: high **{name}** ({score:.2f}) — notable moment' for ts, name, score in peaks]

    summary = ("Overall delivery leaned " + ("warm and engaged." if m["warmth"] >= 60 else "neutral.") +
               (" Dynamic energy helped confidence." if m["vocal_conf"] >= 60 else " Consider adding more dynamic range."))

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
        self.audio_buffer: List[bytes] = []
        self.sample_rate: int = 16000
        self.codec: str = "pcm16"   # 'pcm16' | 'pcm8' | 'opus' | 'opusfs320'
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
# Hume BATCH utilities (REST)
# =========================
def _bytes_to_wav_path(raw: bytes, sample_rate: int) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # PCM16
        wf.setframerate(sample_rate)
        wf.writeframes(raw)
    return tmp_path

def _pcm8_to_pcm16(pcm8: bytes) -> bytes:
    import array
    a = array.array('B', pcm8)                   # unsigned 8-bit
    out = array.array('h', ((x - 128) << 8 for x in a))
    return out.tobytes()

async def hume_measure_batch(audio_bytes: bytes, sample_rate: int) -> Optional[List[Dict[str, Any]]]:
    if not (HUME_API_KEY and http_client):
        print("❌ Hume REST: missing API key or HTTP client"); return None

    wav_path = _bytes_to_wav_path(audio_bytes, sample_rate)
    headers = {"X-Hume-Api-Key": HUME_API_KEY}
    models = {"models": {"prosody": {}}}

    job_id = None
    f = None
    try:
        f = open(wav_path, "rb")
        files = {
            "json": (None, json.dumps(models), "application/json"),
            "file": (os.path.basename(wav_path), f, "audio/wav"),
        }
        resp = await http_client.post("https://api.hume.ai/v0/batch/jobs", headers=headers, files=files, timeout=60.0)
        if resp.status_code // 100 != 2:
            print(f"❌ Hume REST: job submit failed {resp.status_code} {resp.text[:400]}"); return None
        job_id = resp.json().get("job_id")
        if not job_id:
            print(f"❌ Hume REST: job_id missing in response: {resp.text[:300]}"); return None
    except Exception as e:
        print(f"❌ Hume REST: submit error: {e}"); return None
    finally:
        try:
            if f: f.close()
            os.remove(wav_path)
        except Exception:
            pass

    status = "PENDING"
    started = time.time()
    delay = 1.0
    while True:
        try:
            detail = await http_client.get(f"https://api.hume.ai/v0/batch/jobs/{job_id}", headers=headers, timeout=20.0)
            if detail.status_code // 100 == 2:
                j = detail.json()
                status = (j.get("state") or {}).get("status", status)
                if status == "COMPLETED": break
                if status == "FAILED":
                    print(f"❌ Hume REST: job {job_id} failed."); return None
            else:
                print(f"⚠️ Hume REST: status {detail.status_code} {detail.text[:200]}")
        except Exception as e:
            print(f"⚠️ Hume REST: poll error {e}")
        if time.time() - started > 600:
            print(f"❌ Hume REST: job {job_id} timed out."); return None
        await asyncio.sleep(delay); delay = min(delay * 1.5, 6.0)

    try:
        preds = await http_client.get(f"https://api.hume.ai/v0/batch/jobs/{job_id}/predictions",
                                      headers=headers, timeout=60.0)
        if preds.status_code // 100 != 2:
            print(f"❌ Hume REST: predictions fetch failed {preds.status_code} {preds.text[:400]}"); return None
        return preds.json()
    except Exception as e:
        print(f"❌ Hume REST: predictions error {e}")
        return None

# =========================
# Finalization: Build report
# =========================
async def finalize_and_report(st: VoiceState, title_hint="Vocal Tone Session") -> Dict[str, Any]:
    uid = st.uid
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🟥 Finalizing uid={uid}")

    if not st.audio_buffer:
        print(f"⚠️ No audio buffered for uid={uid}. Skipping report.")
        st.active = False; st.started_at = 0.0; st.touch()
        return {"status": "ignored", "reason": "no_audio_buffered"}

    print(f"Combining {len(st.audio_buffer)} audio chunks for batch processing...")
    full_audio_bytes = b"".join(st.audio_buffer)

    # Codec normalization to PCM16 mono
    if st.codec == "pcm8":
        full_audio_bytes = _pcm8_to_pcm16(full_audio_bytes)
    elif st.codec in ("opus", "opusfs320"):
        print("❌ Received Opus but decoder not enabled; please send PCM16 or add Opus decode.")
        st.active = False; st.started_at = 0.0; st.touch()
        return {"status": "error", "reason": "opus_not_supported_server_side"}

    sample_rate = st.sample_rate
    st.audio_buffer = []  # clear before long call

    batch_result_json = await hume_measure_batch(full_audio_bytes, sample_rate)

    if not batch_result_json:
        print(f"❌ Hume Batch job failed or returned empty for uid={uid}.")
        report_text = f'Vocal Tone: -- — "{title_hint}"\nAI Confidence: 0\n\nFailed to process audio with Hume Batch API.'
    else:
        processor = BatchResultProcessor(batch_result_json)
        report_text: Optional[str] = None
        try:
            hume_json_for_llm = processor.compressed_json(topk_per_pred=5, min_score=0.15, max_preds=600)
            sys_prompt = HUME_ANALYSIS_SYSTEM_PROMPT
            user_prompt = build_hume_user_prompt(hume_json_for_llm, title_hint=title_hint)
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
            report_text = await call_llm(messages, temperature=0.2, max_tokens=1400)
        except Exception as e:
            print(f"❌ LLM pipeline error: {e}")
        if not report_text:
            report_text = build_report(processor, title_hint=title_hint)

    await omi_send_notification(uid, title="Your Vocal Tone Report", body=report_text[:800])
    await omi_create_conversation(uid, report_text)
    await omi_save_memory(uid, report_text)

    st.active = False; st.started_at = 0.0; st.touch()
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
    if not hume_client:
        return {"ok": False, "error": "Hume not configured"}
    try:
        await hume_client.expression_measurement.batch.list_jobs(limit=1)
        return {"ok": True, "ping_target": "batch_list_jobs"}
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
    codec: str = Query("pcm16"),  # 'pcm16' | 'pcm8' | 'opus' | 'opusfs320'
):
    key = uid or session_id or request.headers.get("X-Session-ID") or request.headers.get("X-Omi-Uid")
    if not key:
        return {"status": "ignored", "reason": "missing_uid_or_session_id"}

    st = await _get_state(key)
    body = await request.body()
    now = time.time()
    if not body:
        return {"status": "ignored", "reason": "empty_body"}

    async with st.lock:
        if st.active and st.last_wall_ts and (now - st.last_wall_ts > IDLE_TIMEOUT_SEC):
            print(f"⏰ Idle timeout for uid={key} — auto-finalize")
            await finalize_and_report(st)

        if not st.active:
            st.active = True
            st.started_at = now
            st.audio_buffer = []
            st.sample_rate = sample_rate
            st.codec = codec.lower()
            print(f"🟢 session starts (uid={key}, sample_rate={sample_rate}, codec={st.codec})")

        st.touch()
        st.last_audio_ts = now
        st.audio_buffer.append(body)

    return {"status": "ok_buffered", "received": len(body), "sample_rate": sample_rate, "codec": st.codec}

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
    print(f"Starting Hume Vocal Tone server (BATCH MODE) on http://0.0.0.0:{port} (pid={PID})")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
