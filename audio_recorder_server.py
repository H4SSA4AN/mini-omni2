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


@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    audio_file = request.files["file"]
    if audio_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    mime_type = request.form.get("mimeType", "")
    duration = request.form.get("duration", "")

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
        "metrics": {
            "saved_at_ms": saved_at_ms,
            "inference_ms": inference_ms
        }
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
