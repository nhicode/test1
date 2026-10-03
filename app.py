"""Máy chủ nhận dạng nốt piano: nhận NGUYÊN file âm thanh, nhận dạng một lượt (không chia đoạn từ phía trang web).
Mô hình: ByteDance High-resolution Piano Transcription (độ chính xác cao hơn Onsets and Frames)."""
import base64, os, tempfile, threading, time, uuid
import librosa
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from piano_transcription_inference import PianoTranscription

SR = 16000
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None
model_lock = threading.Lock()   # mỗi lúc chỉ chạy 1 bài để không tràn RAM
jobs = {}

def get_model():
    global model
    if model is None:
        model = PianoTranscription(device="cpu")
    return model

def work(job_id, path):
    j = jobs[job_id]
    try:
        with model_lock:
            j["status"] = "running"; j["t0"] = time.time()
            audio, _ = librosa.load(path, sr=SR, mono=True)      # giải mã cả file
            midi_path = path + ".mid"
            res = get_model().transcribe(audio, midi_path)       # nhận dạng cả bài một lượt
            notes = [{"pitch": int(e["midi_note"]), "startTime": float(e["onset_time"]),
                      "endTime": float(e["offset_time"]), "velocity": int(e["velocity"])}
                     for e in res["est_note_events"]]
            with open(midi_path, "rb") as f:
                midi = base64.b64encode(f.read()).decode()
            j["result"] = {"notes": notes, "totalTime": len(audio) / SR, "midi": midi}
            j["status"] = "done"
    except Exception as e:
        j["status"] = "error"; j["error"] = str(e)
    finally:
        for p in (path, path + ".mid"):
            try: os.remove(p)
            except OSError: pass
        j["finished"] = time.time()

@app.get("/")
def health():
    return {"ok": True}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    now = time.time()
    for k in [k for k, v in jobs.items() if v.get("finished") and now - v["finished"] > 600]:
        jobs.pop(k, None)
    suffix = os.path.splitext(file.filename or "")[1] or ".mp3"
    fd, path = tempfile.mkstemp(suffix=suffix); os.close(fd)
    with open(path, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "created": now}
    threading.Thread(target=work, args=(job_id, path), daemon=True).start()
    return {"job": job_id}

@app.get("/job/{job_id}")
def job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        return {"status": "error", "error": "Không tìm thấy công việc (máy chủ có thể đã khởi động lại)."}
    out = {"status": j["status"], "elapsed": int(time.time() - j.get("t0", j["created"]))}
    if j["status"] == "done": out["result"] = j["result"]
    if j["status"] == "error": out["error"] = j["error"]
    return out
