# --------------------------------------------------------------------
# Vocal Tone (Hume STREAM, 4s chunks) -> LLM Report (DeepSeek) -> Push to Omi
# For each POST /audio, forward the whole (≈4s) chunk to Hume stream.
#
# Run (Render Start Command):
#   uvicorn app:app --host 0.0.0.0 --port $PORT
# Client (Omi):
#   POST raw audio to /audio?uid=...&sample_rate=16000&codec=pcm16  (≈4s per request)
#   POST /conversation/end?uid=...  to finalize
# --------------------------------------------------------------------

import os
import io
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

# ---- Hume streaming SDK (pip install "hume[stream]") -------------------------
try:
    from hume.client import AsyncHumeClient
    from hume.expression_measurement.stream import Config as EMConfig
    from hume.expression_measurement.stream.socket_client import StreamConnectOptions
except Exception:
    AsyncHumeClient = None
    EMConfig = None
    StreamConnectOptions = None
# -----------------------------------------------------------------------------

PID = os.getpid()

# =========================
# Env / constants
# =========================
OMI_APP_ID = os.environ.get("OMI_APP_ID")
OMI_APP_SECRET = os.environ.get("OMI_APP_SECRET")  # used for ALL Omi endpoints

HUME_API_KEY = os.environ.get("HUME_API_KEY")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

IDLE_TIMEOUT_SEC = int(os.environ.get("IDLE_TIMEOUT_SEC", "30"))
SUPPORTED_CODECS = {"pcm16", "pcm8"}  # Opus not handled here

# =========================
# App & clients
# =========================
http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🚀 Hume Vocal Tone server (STREAM MODE, 4s) starting...")
    http_client = httpx.AsyncClient(timeout=45.0)
    try:
        yield
    finally:
        if http_client:
            await http_client.aclose()
        print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 👋 Server shutting down...")

app = FastAPI(title="Vocal Tone (Hume Stream 4s x Omi x DeepSeek)", lifespan=lifespan)

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
    params = {"uid": uid, "message": f"{title}: {body}"[:]}
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
Return a concise plain-text "Vocal Tone Report" from Hume Prosody JSON.
Include Overall score, AI Confidence, short highlights, 2 tips, and a breakdown.
Keep it scannable with bullets. Avoid hedging language. Keep under ~2200 chars.
"""

def build_hume_user_prompt(hume_json: dict, title_hint: Optional[str] = None) -> str:
    safe_hint = f"Title hint (optional): {title_hint}" if title_hint else "Title hint (optional):"
    body = json.dumps(hume_json, ensure_ascii=False)
    return f"""{safe_hint}

Hume Prosody JSON:
{body}
"""

# =========================
# Real-time aggregator
# =========================
class RealtimeAgg:
    def __init__(self):
        self.total_dur = 0.0
        self.emotion_sum: Dict[str, float] = {}
        self.peak_events: List[Tuple[float, str, float]] = []

    def ingest(self, result: Dict[str, Any]):
        try:
            models = (result.get("models") or {})
            prosody = models.get("prosody") or {}
            preds = prosody.get("predictions") or []
            for p in preds:
                t = p.get("time") or {}
                begin = float(t.get("begin", 0.0)); end_v = float(t.get("end", begin))
                self.total_dur = max(self.total_dur, end_v)
                for e in (p.get("emotions") or []):
                    name = str(e.get("name", "unknown")).strip()
                    score = float(e.get("score", 0.0))
                    self.emotion_sum[name] = self.emotion_sum.get(name, 0.0) + score
                    if score >= 0.75:
                        self.peak_events.append((begin, name, score))
        except Exception as e:
            print(f"⚠️ RealtimeAgg ingest error: {e}. Result preview: {json.dumps(result)[:500]}")

    def compact(self):
        return {
            "meta": {"approx_audio_duration_sec": self.total_dur},
            "prosody_summary": {
                "top_emotions": sorted(
                    [{"name": n, "sum_score": s} for n, s in self.emotion_sum.items()],
                    key=lambda x: x["sum_score"], reverse=True
                )[:10],
                "peak_events": [{"t": t, "name": n, "score": s} for (t, n, s) in self.peak_events[:10]],
            },
        }

# =========================
# Deterministic scoring (fallback)
# =========================
POSITIVE = {"Admiration","Amusement","Awe","Calmness","Contentment","Interest","Joy","Relief","Triumph","Tenderness"}
TENSION  = {"Annoyance","Anxiety","Distress","Frustration","Irritation","Sadness","Disappointment","Contempt","Anger","Doubt"}
ENERGY   = {"Excitement","Elation","Determination","Surprise"}
ENGAGE   = {"Interest","Curiosity","Concentration","Enthusiasm","Attentiveness"}

def _avg_over(total_dur: float, emotion_sum: Dict[str, float], names: set) -> float:
    if total_dur <= 0: return 0.0
    s = sum(emotion_sum.get(n, 0.0) for n in names)
    return 100.0 * (s / max(total_dur, 1e-6))

def compute_metrics(total_dur: float, emotion_sum: Dict[str, float], peak_events: List[Tuple[float,str,float]]) -> dict:
    warmth   = max(0, min(100, int(_avg_over(total_dur, emotion_sum, POSITIVE))))
    tension  = max(0, min(100, int(_avg_over(total_dur, emotion_sum, TENSION))))
    energy   = max(0, min(100, int(_avg_over(total_dur, emotion_sum, ENERGY))))
    engage   = max(0, min(100, int(_avg_over(total_dur, emotion_sum, ENGAGE))))
    empathy  = 50
    if peak_events:
        has_tension  = any(n in TENSION for _, n, _ in peak_events)
        has_positive = any(n in POSITIVE for _, n, _ in peak_events)
        empathy = 70 if (has_tension and has_positive) else (60 if has_positive else 50)
    vocal_conf = max(0, min(100, int(0.6*energy + 0.4*(100 - tension))))
    vocal_tone = max(0, min(100, int(0.5*warmth + 0.3*vocal_conf + 0.2*engage - 0.1*tension)))
    ai_conf    = min(95, int(20 + 75 * min(1.0, total_dur / 60.0)))
    return {"warmth": warmth, "tension": tension, "energy": energy, "engage": engage,
            "vocal_conf": vocal_conf, "vocal_tone": vocal_tone, "ai_conf": ai_conf}

def build_report_from_agg(agg: RealtimeAgg, title_hint: str = "Vocal Tone Session") -> str:
    m = compute_metrics(agg.total_dur, agg.emotion_sum, agg.peak_events)
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
# Hume streaming session (forward whole 4s chunk)
# =========================
def _pcm8_to_pcm16(pcm8: bytes) -> bytes:
    import array
    a = array.array('B', pcm8)
    out = array.array('h', ((x - 128) << 8 for x in a))
    return out.tobytes()

def _wav_from_pcm16(pcm16: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return buf.getvalue()

class HumeStreamSession:
    def __init__(self, uid: str, sample_rate: int):
        self.uid = uid
        self.sample_rate = int(sample_rate)
        self.last_wall_ts = time.time()
        self.active = False
        self._client: Optional[AsyncHumeClient] = None
        self._socket_cm = None
        self._socket = None
        self.agg = RealtimeAgg()

    async def start(self):
        if not (AsyncHumeClient and EMConfig and StreamConnectOptions and HUME_API_KEY):
            raise RuntimeError("Hume streaming not configured.")
        self._client = AsyncHumeClient(api_key=HUME_API_KEY)
        self._socket_cm = self._client.expression_measurement.stream.connect(
            options=StreamConnectOptions(config=EMConfig(prosody={}), interim_results=True)
        )
        self._socket = await self._socket_cm.__aenter__()
        self.active = True
        self.touch()
        print(f"🟢 Hume stream opened (uid={self.uid}, fs={self.sample_rate}Hz)")

    async def close(self):
        try:
            if self._socket_cm:
                await self._socket_cm.__aexit__(None, None, None)
        except Exception as e:
            print(f"⚠️ Hume stream close error: {e}")
        self.active = False
        print(f"🔴 Hume stream closed (uid={self.uid})")

    def touch(self):
        self.last_wall_ts = time.time()

    async def send_chunk(self, raw: bytes, codec: str):
        if not self.active:
            await self.start()
        self.touch()

        if codec == "pcm8":
            raw = _pcm8_to_pcm16(raw)
        elif codec != "pcm16":
            raise ValueError(f"Unsupported codec={codec}; send pcm16 or pcm8.")

        # Optional: sanity-check inferred duration (defensive)
        # dur_sec = len(raw) / (2 * self.sample_rate)
        # if dur_sec > 5.1: raise ValueError(f"Chunk too long ({dur_sec:.2f}s). Max 5s per message.")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            wav_bytes = _wav_from_pcm16(raw, self.sample_rate)
            tmp.write(wav_bytes); tmp.flush(); tmp.close()
            result = await self._socket.send_file(file=tmp.name)
            if isinstance(result, dict):
                self.agg.ingest(result)
        except Exception as e:
            print(f"❌ Hume send_chunk error: {e}")
        finally:
            try: os.remove(tmp.name)
            except Exception: pass

# =========================
# Per-UID state
# =========================
class VoiceState:
    def __init__(self, uid: str):
        self.uid = uid
        self.session: Optional[HumeStreamSession] = None
        self.sample_rate: int = 16000
        self.codec: str = "pcm16"
        self.started_at = 0.0
        self.active = False
        self.lock = asyncio.Lock()

    def touch(self):
        if self.session:
            self.session.touch()

STATES: Dict[str, VoiceState] = {}
STATES_LOCK = asyncio.Lock()

async def _get_state(uid: str) -> VoiceState:
    async with STATES_LOCK:
        if uid not in STATES:
            STATES[uid] = VoiceState(uid)
        return STATES[uid]

# =========================
# Finalization
# =========================
async def finalize_and_report(st: VoiceState, title_hint="Vocal Tone Session") -> Dict[str, Any]:
    uid = st.uid
    print(f"[{datetime.now(timezone.utc).isoformat()}] (pid={PID}) 🟥 Finalizing uid={uid}")

    if not st.session or not st.session.active:
        print(f"⚠️ No active stream for uid={uid}.")
        return {"status": "ignored", "reason": "no_active_stream"}

    agg = st.session.agg
    report_text: Optional[str] = None

    try:
        hume_json_for_llm = agg.compact()
        sys_prompt = HUME_ANALYSIS_SYSTEM_PROMPT
        user_prompt = build_hume_user_prompt(hume_json_for_llm, title_hint=title_hint)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
        report_text = await call_llm(messages, temperature=0.2, max_tokens=1400)
    except Exception as e:
        print(f"❌ LLM pipeline error: {e}")

    if not report_text:
        report_text = build_report_from_agg(agg, title_hint=title_hint)

    await omi_send_notification(uid, title="Your Vocal Tone Report", body=report_text[:800])
    await omi_create_conversation(uid, report_text)
    await omi_save_memory(uid, report_text)

    try:
        await st.session.close()
    except Exception:
        pass

    st.active = False
    st.started_at = 0.0
    return {"status": "success", "summary": {"report": report_text}}

# =========================
# Endpoints
# =========================
@app.get("/")
async def health():
    return {
        "status": "ok",
        "omi_creds_loaded": bool(OMI_APP_ID and OMI_APP_SECRET),
        "hume_ready": bool(HUME_API_KEY and AsyncHumeClient and EMConfig and StreamConnectOptions),
        "llm_ready": bool(DEEPSEEK_API_KEY),
        "pid": PID,
        "mode": "stream_forward_4s",
    }

@app.get("/hume/ping")
async def hume_ping():
    if not (HUME_API_KEY and AsyncHumeClient and EMConfig and StreamConnectOptions):
        return {"ok": False, "error": "Hume streaming not configured"}
    try:
        client = AsyncHumeClient(api_key=HUME_API_KEY)
        async with client.expression_measurement.stream.connect(
            options=StreamConnectOptions(config=EMConfig(prosody={}))
        ) as socket:
            jd = await socket.get_job_details()
            return {"ok": True, "ping_target": "stream.connect", "job_details": bool(jd)}
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
    codec: str = Query("pcm16"),  # 'pcm16' | 'pcm8'
):
    key = uid or session_id or request.headers.get("X-Session-ID") or request.headers.get("X-Omi-Uid")
    if not key:
        return {"status": "ignored", "reason": "missing_uid_or_session_id"}

    raw = await request.body()
    if not raw:
        return {"status": "ignored", "reason": "empty_body"}

    if codec.lower() not in SUPPORTED_CODECS:
        return {"status": "error", "reason": f"unsupported_codec:{codec}"}

    st = await _get_state(key)
    async with st.lock:
        # idle auto-finalize
        if st.active and st.session and (time.time() - st.session.last_wall_ts > IDLE_TIMEOUT_SEC):
            print(f"⏰ Idle timeout for uid={key} — auto-finalize")
            return await finalize_and_report(st)

        # start session on first audio
        if not st.active or not st.session:
            st.sample_rate = sample_rate
            st.codec = codec.lower()
            st.session = HumeStreamSession(key, sample_rate=sample_rate)
            await st.session.start()
            st.active = True
            st.started_at = time.time()
            print(f"🟢 session starts (uid={key}, fs={sample_rate}, codec={st.codec})")

        try:
            await st.session.send_chunk(raw, st.codec)  # forward entire ~4s chunk
        except Exception as e:
            return {"status": "error", "reason": f"send_chunk_failed:{e}"}

    return {"status": "ok_forwarded", "received": len(raw), "sample_rate": sample_rate, "codec": st.codec}

@app.post("/conversation/end")
async def conversation_end(uid: Optional[str] = Query(None), session_id: Optional[str] = Query(None)):
    key = uid or session_id
    if not key:
        return {"status": "ignored", "reason": "missing_uid_or_session_id"}
    st = await _get_state(key)
    async with st.lock:
        if not st.active or not st.session:
            return {"status": "ignored", "reason": "not_active"}
        return await finalize_and_report(st)

# Entrypoint
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Hume Vocal Tone server (STREAM MODE, 4s) on http://0.0.0.0:{port} (pid={PID})")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
