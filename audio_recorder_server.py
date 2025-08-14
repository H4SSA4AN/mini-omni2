#!/usr/bin/env python3
"""
Audio Recorder Server (Flask)
- Serves a simple web page that records audio in the browser
- Saves the recorded audio to the 'recordings' folder when user presses Stop
- Keeps only ONE file in 'recordings' at any time and names it 'UserInput.wav'
- Converts incoming audio (e.g., webm/ogg) to WAV using pydub/ffmpeg when needed
- After saving WAV, runs Mini Omni 2 A1A2 inference and writes answer to 'answers/Answer.wav'
- Serves recordings and answers with no-cache headers so the browser always fetches the latest
"""

import os
import datetime
import time
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, abort
from flask_cors import CORS
import requests

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    PYDUB_AVAILABLE = False

# Mini Omni 2 imports
import torch
from inference import (
    load_model as omni_load_model,
    load_audio as omni_load_audio,
    get_input_ids_whisper as omni_get_input_ids_whisper,
    A1_A2 as omni_A1_A2,
)
import shutil

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

BASE_DIR = Path(__file__).parent
RECORDINGS_DIR = BASE_DIR / "recordings"
ANSWERS_DIR = BASE_DIR / "answers"
RECORDINGS_DIR.mkdir(exist_ok=True)
ANSWERS_DIR.mkdir(exist_ok=True)

# Globals for Mini Omni 2 model
OMNI_INITIALIZED = False
OMNI_DEVICE = None
OMNI_FABRIC = None
OMNI_MODEL = None
OMNI_TOKENIZER = None
OMNI_SNAC = None
OMNI_WHISPER = None
OMNI_CKPT = str(BASE_DIR / "checkpoint")


def _clear_dir(dir_path: Path) -> None:
    try:
        for item in dir_path.iterdir():
            if item.is_file():
                try:
                    item.unlink(missing_ok=True)
                except TypeError:
                    if item.exists():
                        item.unlink()
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
    except Exception as e:
        print(f"⚠️ Failed to clear {dir_path}: {e}")


def _init_omni() -> None:
    global OMNI_INITIALIZED, OMNI_DEVICE, OMNI_FABRIC, OMNI_MODEL, OMNI_TOKENIZER, OMNI_SNAC, OMNI_WHISPER
    if OMNI_INITIALIZED:
        return
    try:
        OMNI_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Initializing Mini Omni 2 on device: {OMNI_DEVICE}")
        OMNI_FABRIC, OMNI_MODEL, OMNI_TOKENIZER, OMNI_SNAC, OMNI_WHISPER = omni_load_model(OMNI_CKPT, OMNI_DEVICE)
        OMNI_INITIALIZED = True
        print("✅ Mini Omni 2 initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Mini Omni 2: {e}")
        raise


def _normalize_musetalk_upload_url(url: str) -> str:
    try:
        from urllib.parse import urlparse, urlunparse
        p = urlparse(url)
        # If no scheme, assume http
        scheme = p.scheme or 'http'
        netloc = p.netloc or p.path  # handle bare host without scheme
        path = '/upload_answer'
        return urlunparse((scheme, netloc, path, '', '', ''))
    except Exception:
        return url.rstrip('/') + '/upload_answer'

def _normalize_musetalk_start_url(url: str) -> str:
    try:
        from urllib.parse import urlparse, urlunparse
        p = urlparse(url)
        scheme = p.scheme or 'http'
        netloc = p.netloc or p.path
        path = '/start'
        return urlunparse((scheme, netloc, path, '', '', ''))
    except Exception:
        return url.rstrip('/') + '/start'


def _forward_answer_and_trigger_inference(answer_path: Path, musetalk_url: str, fps: int, batch_size: int) -> dict:
    """
    1. POST the generated Answer.wav to MuseTalk server's /upload_answer.
    2. POST to MuseTalk server's /start to trigger inference with custom params.
    """
    try:
        # Step 1: Upload the audio file
        upload_url = _normalize_musetalk_upload_url(musetalk_url)
        files = {'file': ('Answer.wav', open(answer_path, 'rb'), 'audio/wav')}
        upload_resp = requests.post(upload_url, files=files, timeout=10)
        
        if not upload_resp.ok:
            return {"ok": False, "step": "upload", "status": upload_resp.status_code, "text": upload_resp.text}
        
        # Step 2: Trigger inference with parameters
        start_url = _normalize_musetalk_start_url(musetalk_url)
        payload = {"fps": fps, "batch_size": batch_size}
        start_resp = requests.post(start_url, json=payload, timeout=10)

        if not start_resp.ok:
            return {"ok": False, "step": "start", "status": start_resp.status_code, "text": start_resp.text}
            
        return {"ok": True, "upload_response": upload_resp.json(), "start_response": start_resp.json()}

    except Exception as e:
        return {"ok": False, "error": str(e)}


# Clear the recordings folder on startup
_clear_dir(RECORDINGS_DIR)


@app.route("/health")
def health() -> tuple[dict, int]:
    return {
        "status": "ok",
        "recordings_dir": str(RECORDINGS_DIR),
        "answers_dir": str(ANSWERS_DIR),
        "pydub": PYDUB_AVAILABLE,
        "omni_initialized": OMNI_INITIALIZED,
        "device": OMNI_DEVICE or "unknown",
    }, 200


@app.route("/")
def index():
    return render_template("recorder.html")


def _run_a1a2_inference(user_wav: Path) -> dict:
    _init_omni()
    _clear_dir(ANSWERS_DIR)
    mel, leng = omni_load_audio(str(user_wav))
    audio_feature, input_ids = omni_get_input_ids_whisper(mel, leng, OMNI_WHISPER, OMNI_DEVICE)
    temp_out_dir = str(ANSWERS_DIR)
    text_response = omni_A1_A2(
        OMNI_FABRIC,
        audio_feature,
        input_ids,
        leng,
        OMNI_MODEL,
        OMNI_TOKENIZER,
        0,
        OMNI_SNAC,
        out_dir=temp_out_dir,
    )
    generated_path = ANSWERS_DIR / "A1-A2" / "00.wav"
    final_answer = ANSWERS_DIR / "Answer.wav"
    if generated_path.exists():
        try:
            shutil.move(str(generated_path), str(final_answer))
            try:
                (ANSWERS_DIR / "A1-A2").rmdir()
            except Exception:
                pass
        except Exception as e:
            raise RuntimeError(f"Failed to finalize answer file: {e}")
    else:
        raise FileNotFoundError(f"Generated answer not found at {generated_path}")
    return {
        "text_response": text_response,
        "answer_path": str(final_answer),
        "answer_url": "/answers/Answer.wav",
    }


@app.route("/check_musetalk", methods=["POST"])
def check_musetalk():
    try:
        data = request.get_json(silent=True) or {}
        base_url = (data.get("url") or "").strip()
        if not base_url:
            return jsonify({"ok": False, "error": "missing url"}), 400

        # Candidates to try: exact URL, URL + '/health', URL root
        candidates = []
        def add(url: str):
            if url and url not in [c["url"] for c in candidates]:
                candidates.append({"url": url})
        add(base_url)
        if not base_url.rstrip("/").endswith("health"):
            add(base_url.rstrip("/") + "/health")
        # try root if a path exists
        if "/" in base_url.rstrip("/")[8:]:  # after scheme
            root = base_url.split("//", 1)[-1]
            root = root.split("/", 1)[0]
            scheme = "http" if base_url.lower().startswith("http://") else "https" if base_url.lower().startswith("https://") else "http"
            add(f"{scheme}://{root}")

        timeout = float(data.get("timeout", 2.5))
        tried = []
        for c in candidates:
            url = c["url"]
            try:
                resp = requests.get(url, timeout=timeout)
                tried.append({"url": url, "status": resp.status_code})
                if resp.ok:
                    return jsonify({
                        "ok": True,
                        "url": url,
                        "status": resp.status_code,
                        "elapsed_ms": int(resp.elapsed.total_seconds() * 1000),
                        "tried": tried,
                    }), 200
            except Exception as e:
                tried.append({"url": url, "status": None, "error": str(e)})

        return jsonify({
            "ok": False,
            "tried": tried,
        }), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    audio_file = request.files["file"]
    if audio_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    mime_type = request.form.get("mimeType", "")
    duration = request.form.get("duration", "")
    musetalk_url = request.form.get("musetalkUrl", "").strip()
    try:
        fps = int(request.form.get("fps", 25))
        batch_size = int(request.form.get("batch_size", 8))
    except (ValueError, TypeError):
        fps = 25
        batch_size = 8

    _clear_dir(RECORDINGS_DIR)
    wav_path = RECORDINGS_DIR / "UserInput.wav"

    is_wav_upload = (mime_type.lower() == "audio/wav") or audio_file.filename.lower().endswith(".wav")
    if is_wav_upload:
        audio_file.save(wav_path)
    else:
        if not PYDUB_AVAILABLE:
            return jsonify({
                "error": "Conversion to WAV requires pydub. Please install pydub and ffmpeg.",
                "hint": "pip install pydub and ensure ffmpeg is in PATH"
            }), 500
        temp_ext = audio_file.filename.split(".")[-1].lower() if "." in audio_file.filename else "webm"
        temp_input = RECORDINGS_DIR / f"UserInput_input.{temp_ext}"
        audio_file.save(temp_input)
        try:
            seg = AudioSegment.from_file(temp_input)
            seg = seg.set_channels(1).set_frame_rate(24000)
            seg.export(wav_path, format="wav")
        except Exception as e:
            return jsonify({
                "error": f"Failed to convert to WAV: {e}",
                "hint": "Ensure ffmpeg is installed and available in PATH"
            }), 500
        finally:
            try:
                temp_input.unlink(missing_ok=True)
            except TypeError:
                if temp_input.exists():
                    temp_input.unlink()

    file_size = wav_path.stat().st_size if wav_path.exists() else 0

    # Timing metrics
    saved_at_ms = int(time.time() * 1000)
    infer_start = time.time()
    try:
        inference_result = _run_a1a2_inference(wav_path)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500
    inference_ms = int((time.time() - infer_start) * 1000)

    # Optionally forward the answer to MuseTalk server if URL provided by client
    forward_info = None
    if musetalk_url:
        try:
            forward_info = _forward_answer_and_trigger_inference(
                Path(inference_result["answer_path"]), 
                musetalk_url,
                fps,
                batch_size
            )
        except Exception as e:
            forward_info = {"ok": False, "error": str(e)}

    return jsonify({
        "status": "saved",
        "user_input": {
            "filename": "UserInput.wav",
            "file_path": str(wav_path),
            "file_url": "/recordings/UserInput.wav",
            "mime_type": "audio/wav",
            "duration": duration,
            "size_bytes": file_size,
        },
        "answer": inference_result,
        "forwarded_to_musetalk": forward_info,
        "metrics": {
            "saved_at_ms": saved_at_ms,
            "inference_ms": inference_ms
        }
    }), 200


@app.route("/upload_raw_audio", methods=["POST"])
def upload_raw_audio():
    """Upload raw audio and send directly to musetalk without omni inference"""
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    audio_file = request.files["file"]
    if audio_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    mime_type = request.form.get("mimeType", "")
    duration = request.form.get("duration", "")
    musetalk_url = request.form.get("musetalkUrl", "").strip()
    try:
        fps = int(request.form.get("fps", 25))
        batch_size = int(request.form.get("batch_size", 8))
    except (ValueError, TypeError):
        fps = 25
        batch_size = 8

    if not musetalk_url:
        return jsonify({"error": "MuseTalk URL is required for raw audio recording"}), 400

    _clear_dir(ANSWERS_DIR)
    wav_path = ANSWERS_DIR / "Answer.wav"

    is_wav_upload = (mime_type.lower() == "audio/wav") or audio_file.filename.lower().endswith(".wav")
    if is_wav_upload:
        audio_file.save(wav_path)
    else:
        if not PYDUB_AVAILABLE:
            return jsonify({
                "error": "Conversion to WAV requires pydub. Please install pydub and ffmpeg.",
                "hint": "pip install pydub and ensure ffmpeg is in PATH"
            }), 500
        temp_ext = audio_file.filename.split(".")[-1].lower() if "." in audio_file.filename else "webm"
        temp_input = ANSWERS_DIR / f"temp_input.{temp_ext}"
        audio_file.save(temp_input)
        try:
            seg = AudioSegment.from_file(temp_input)
            seg = seg.set_channels(1).set_frame_rate(24000)
            seg.export(wav_path, format="wav")
        except Exception as e:
            return jsonify({
                "error": f"Failed to convert to WAV: {e}",
                "hint": "Ensure ffmpeg is installed and available in PATH"
            }), 500
        finally:
            try:
                temp_input.unlink(missing_ok=True)
            except TypeError:
                if temp_input.exists():
                    temp_input.unlink()

    file_size = wav_path.stat().st_size if wav_path.exists() else 0

    # Forward the raw audio directly to MuseTalk server
    forward_info = None
    try:
        forward_info = _forward_answer_and_trigger_inference(
            wav_path, 
            musetalk_url,
            fps,
            batch_size
        )
    except Exception as e:
        forward_info = {"ok": False, "error": str(e)}

    return jsonify({
        "status": "saved",
        "raw_audio": {
            "filename": "Answer.wav",
            "file_path": str(wav_path),
            "file_url": "/answers/Answer.wav",
            "mime_type": "audio/wav",
            "duration": duration,
            "size_bytes": file_size,
        },
        "forwarded_to_musetalk": forward_info,
        "note": "Raw audio sent directly to MuseTalk (no omni inference)"
    }), 200


@app.route("/recordings/<path:filename>")
def serve_recording(filename: str):
    target = RECORDINGS_DIR / filename
    if not target.exists() or not target.is_file():
        abort(404)
    resp = send_from_directory(RECORDINGS_DIR, filename)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/answers/<path:filename>")
def serve_answer(filename: str):
    target = ANSWERS_DIR / filename
    if not target.exists() or not target.is_file():
        abort(404)
    resp = send_from_directory(ANSWERS_DIR, filename)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/musetalk_stream_ready", methods=["POST"])
def musetalk_stream_ready():
    """Receive notification from MuseTalk that streaming is ready"""
    try:
        data = request.get_json()
        if data and data.get("status") == "ready":
            stream_url = data.get("stream_url")
            print(f"[Mini-Omni] MuseTalk stream ready: {stream_url}")
            return jsonify({"ok": True, "message": "Stream notification received"})
        else:
            return jsonify({"ok": False, "error": "Invalid notification data"}), 400
    except Exception as e:
        print(f"[Mini-Omni] Error handling stream notification: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500





@app.route("/musetalk_webrtc_offer", methods=["POST"])
def musetalk_webrtc_offer():
    """Proxy an SDP offer to a MuseTalk WebRTC server to avoid browser CORS issues.

    Request JSON:
      {
        "offer_url": "http://localhost:8090/offer",
        "sdp": "...",
        "type": "offer"
      }
    """
    try:
        data = request.get_json(silent=True) or {}
        offer_url = (data.get("offer_url") or "").strip()
        sdp = data.get("sdp")
        sdp_type = data.get("type") or "offer"
        if not offer_url:
            return jsonify({"ok": False, "error": "offer_url required"}), 400
        if not sdp:
            return jsonify({"ok": False, "error": "sdp required"}), 400
        # Forward
        try:
            resp = requests.post(offer_url, json={"sdp": sdp, "type": sdp_type}, timeout=10)
        except Exception as e:
            return jsonify({"ok": False, "error": f"request error: {e}"}), 502
        if not resp.ok:
            return jsonify({"ok": False, "status": resp.status_code, "text": resp.text}), 502
        try:
            payload = resp.json()
        except Exception:
            payload = {"sdp": resp.text, "type": "answer"}
        return jsonify({"ok": True, "answer": payload}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
