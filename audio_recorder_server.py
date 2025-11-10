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
import base64
import json
import logging
import socket
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
UPLOADS_DIR = BASE_DIR / "uploads"
RECORDINGS_DIR.mkdir(exist_ok=True)
ANSWERS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Globals for Mini Omni 2 model
OMNI_INITIALIZED = False
OMNI_DEVICE = None
OMNI_FABRIC = None
OMNI_MODEL = None
OMNI_TOKENIZER = None
OMNI_SNAC = None
OMNI_WHISPER = None
OMNI_CKPT = str(BASE_DIR / "checkpoint")

# Config and state (integrated from aio_app.py)
MAX_AUDIO_SIZE = 100 * 1024 * 1024  # 100MB
MUSETALK_URL = os.getenv('MUSETALK_URL', 'http://localhost:8085')

# Simple in-memory frame buffer (for NDJSON frames posted by MuseTalk)
frame_buffer = []
processing_complete = False
start_signal_received = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    POST the generated Answer.wav to MuseTalk server's /process with required multipart form fields.
    Fields: audio (file), stream_url (callback), fps, batch_size, bbox_shift.
    """
    try:
        # Normalize the base URL (remove any paths like /stream, /health, etc.)
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(musetalk_url)
        base_url = urlunparse((parsed.scheme or 'http', parsed.netloc, '', '', '', ''))
        print(f"Original musetalk_url: {musetalk_url}")
        print(f"Normalized base_url: {base_url}")

        # Build process URL
        process_url = f"{base_url}/process"

        # Build a public callback URL for MuseTalk to POST frames back to this app
        xf_proto = request.headers.get('X-Forwarded-Proto')
        xf_host = request.headers.get('X-Forwarded-Host')
        scheme = xf_proto or request.scheme
        host = xf_host or request.host
        stream_url = f"{scheme}://{host}/stream_frames"

        # Prepare multipart form-data and send
        print(f"Process URL: {process_url}")
        print(f"Callback stream_url: {stream_url}")
        with open(answer_path, 'rb') as fh:
            files = {
                # Use actual filename and content type based on extension
                'audio': (
                    answer_path.name,
                    fh,
                    'audio/mpeg' if answer_path.suffix.lower() == '.mp3' else 'audio/wav'
                )
            }
            data = {
                'stream_url': stream_url,
                'fps': str(int(fps)),
                'batch_size': str(int(batch_size)),
                'bbox_shift': '0'
            }
            resp = requests.post(process_url, files=files, data=data, timeout=120)

            if not resp.ok:
                return {"ok": False, "step": "process", "status": resp.status_code, "text": resp.text}

            # Try to parse JSON response; fall back to text
            try:
                payload = resp.json()
            except Exception:
                payload = {"text": resp.text}
            return {"ok": True, "process_response": payload}

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


@app.route("/save_audio", methods=["POST"])
def save_audio_handler():
    """Accept base64 audio JSON and forward processed audio to MuseTalk /process."""
    try:
        if request.content_length and request.content_length > MAX_AUDIO_SIZE:
            return jsonify({'error': 'Request too large'}), 413

        data = request.get_json(silent=True) or {}
        audio_data = data.get('audio_data')
        fps = str(data.get('fps', '25'))
        batch_size = str(data.get('batch_size', '20'))
        musetalk_base_url = MUSETALK_URL
        mode = (data.get('mode') or 'pipeline').strip().lower()

        if not audio_data:
            return jsonify({'error': 'No audio data received'}), 400

        if isinstance(audio_data, str) and audio_data.startswith('data:audio/wav;base64,'):
            audio_data = audio_data.split(',', 1)[1]

        try:
            audio_bytes = base64.b64decode(audio_data)
        except Exception as e:
            return jsonify({'error': f'Invalid audio data: {e}'}), 400

        if len(audio_bytes) > MAX_AUDIO_SIZE:
            return jsonify({'error': 'Audio file too large'}), 413

        input_wav_path = UPLOADS_DIR / 'input.wav'
        with open(input_wav_path, 'wb') as f:
            f.write(audio_bytes)
        saved_at = datetime.datetime.now().isoformat()

        # Prepare answer audio (try MP3 via pydub; fallback to WAV)
        answer_path = UPLOADS_DIR / 'answer.mp3'
        answer_filename = 'answer.mp3'
        answer_content_type = 'audio/mpeg'
        try:
            if PYDUB_AVAILABLE and input_wav_path.exists():
                seg = AudioSegment.from_file(input_wav_path)
                seg = seg.set_channels(1).set_frame_rate(24000)
                try:
                    seg.export(answer_path, format="mp3")
                except Exception:
                    answer_path = input_wav_path
                    answer_filename = 'input.wav'
                    answer_content_type = 'audio/wav'
            else:
                answer_path = input_wav_path
                answer_filename = 'input.wav'
                answer_content_type = 'audio/wav'
        except Exception:
            answer_path = input_wav_path
            answer_filename = 'input.wav'
            answer_content_type = 'audio/wav'

        # Build MuseTalk /process endpoint
        musetalk_base_url = str(musetalk_base_url).strip()
        if musetalk_base_url.endswith('/'):
            musetalk_base_url = musetalk_base_url[:-1]
        if not (musetalk_base_url.startswith('http://') or musetalk_base_url.startswith('https://')):
            musetalk_base_url = 'http://' + musetalk_base_url
        musetalk_url = musetalk_base_url + '/process'

        # Build callback URL
        xf_proto = request.headers.get('X-Forwarded-Proto')
        xf_host = request.headers.get('X-Forwarded-Host')
        scheme = xf_proto or request.scheme
        host = xf_host or request.host
        stream_url = f"{scheme}://{host}/stream_frames"

        files = {
            'audio': (answer_filename, open(answer_path, 'rb'), answer_content_type)
        }
        form = {
            'stream_url': stream_url,
            'fps': fps,
            'batch_size': batch_size,
            'bbox_shift': '0'
        }

        resp = None
        try:
            resp = requests.post(musetalk_url, files=files, data=form, timeout=120)
            musetalk_ok = resp.ok
            try:
                musetalk_payload = resp.json()
            except Exception:
                musetalk_payload = {'text': resp.text if resp is not None else ''}
        finally:
            try:
                files['audio'][1].close()
            except Exception:
                pass

        return jsonify({
            'success': musetalk_ok,
            'message': 'Answer audio forwarded to MuseTalk',
            'musetalk_response': musetalk_payload,
            'stream_url': stream_url,
            'saved_at': saved_at,
            'transcript': '',
            'answer': '',
            'answer_audio_path': str(answer_path),
            'answer_audio_url': f"{scheme}://{host}/uploads/{answer_filename}",
        }), 200 if musetalk_ok else 502

    except Exception as e:
        print(f"[Mini-Omni] save_audio error: {e}")
        return jsonify({'error': str(e)}), 500

def _run_a1a2_inference(user_wav: Path) -> dict:
    """Run Mini-Omni inference and write Answer.wav into ANSWERS_DIR."""
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
    """Directly check MuseTalk health without proxy; returns status and diagnostics."""
    try:
        data = request.get_json(silent=True) or {}
        base_url = (data.get("url") or "").strip()
        if not base_url:
            return jsonify({"ok": False, "error": "missing url"}), 400

        # Normalize base URL
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(base_url)
        musetalk_base_url = urlunparse((parsed.scheme or 'http', parsed.netloc or parsed.path, '', '', '', ''))

        # Build health URL with this app's public base
        xf_proto = request.headers.get('X-Forwarded-Proto')
        xf_host = request.headers.get('X-Forwarded-Host')
        scheme = xf_proto or request.scheme
        host = xf_host or request.host
        public_base = f"{scheme}://{host}"
        url = musetalk_base_url.rstrip('/') + '/health' + f"?stream_base={public_base}"

        # Perform GET
        try:
            resp = requests.get(url, headers={'User-Agent': 'AvatarPageProbe/1.0'}, timeout=5)
            ct = resp.headers.get('Content-Type', '')
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            return jsonify({
                'ok': resp.status_code == 200,
                'status': resp.status_code,
                'url': url,
                'content_type': ct,
                'body': body,
            }), 200 if resp.status_code == 200 else 502
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e), 'url': url}), 504

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
    musetalk_url = (request.form.get("musetalkUrl", "") or MUSETALK_URL).strip()
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
            # Convert Omni Answer.wav to MP3 for MuseTalk, fallback to WAV
            src_wav = Path(inference_result["answer_path"])
            mp3_path = ANSWERS_DIR / "Answer.mp3"
            use_path = src_wav
            if PYDUB_AVAILABLE and src_wav.exists():
                try:
                    seg = AudioSegment.from_file(src_wav)
                    seg = seg.set_channels(1).set_frame_rate(24000)
                    seg.export(mp3_path, format="mp3")
                    use_path = mp3_path
                except Exception:
                    use_path = src_wav
            forward_info = _forward_answer_and_trigger_inference(
                use_path,
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
    musetalk_url = (request.form.get("musetalkUrl", "") or MUSETALK_URL).strip()
    try:
        fps = int(request.form.get("fps", 25))
        batch_size = int(request.form.get("batch_size", 8))
    except (ValueError, TypeError):
        fps = 25
        batch_size = 8

    if not musetalk_url:
        return jsonify({"error": "MuseTalk URL is required for raw audio recording"}), 400

    _clear_dir(ANSWERS_DIR)
    mp3_path = ANSWERS_DIR / "Answer.mp3"

    file_ext = audio_file.filename.split(".")[-1].lower() if "." in audio_file.filename else ""
    is_mp3_upload = (mime_type.lower() == "audio/mpeg") or audio_file.filename.lower().endswith(".mp3")
    if is_mp3_upload:
        audio_file.save(mp3_path)
    else:
        # Convert input to MP3 if possible
        if not PYDUB_AVAILABLE:
            return jsonify({
                "error": "Conversion to MP3 requires pydub. Please install pydub and ffmpeg.",
                "hint": "pip install pydub and ensure ffmpeg is in PATH"
            }), 500
        temp_input = ANSWERS_DIR / f"temp_input.{file_ext or 'bin'}"
        audio_file.save(temp_input)
        try:
            seg = AudioSegment.from_file(temp_input)
            seg = seg.set_channels(1).set_frame_rate(24000)
            seg.export(mp3_path, format="mp3")
        except Exception as e:
            return jsonify({
                "error": f"Failed to convert to MP3: {e}",
                "hint": "Ensure ffmpeg is installed and available in PATH"
            }), 500
        finally:
            try:
                temp_input.unlink(missing_ok=True)
            except TypeError:
                if temp_input.exists():
                    temp_input.unlink()

    file_size = mp3_path.stat().st_size if mp3_path.exists() else 0

    # Forward the raw audio directly to MuseTalk server
    forward_info = None
    try:
        forward_info = _forward_answer_and_trigger_inference(
            mp3_path, 
            musetalk_url,
            fps,
            batch_size
        )
    except Exception as e:
        forward_info = {"ok": False, "error": str(e)}

    return jsonify({
        "status": "saved",
        "raw_audio": {
            "filename": "Answer.mp3",
            "file_path": str(mp3_path),
            "file_url": "/answers/Answer.mp3",
            "mime_type": "audio/mpeg",
            "duration": duration,
            "size_bytes": file_size,
        },
        "forwarded_to_musetalk": forward_info,
        "note": "Raw audio sent directly to MuseTalk (no omni inference)"
    }), 200


@app.route("/upload_audio_file", methods=["POST"])
def upload_audio_file():
    """Upload an audio file and send directly to musetalk without omni inference"""
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    audio_file = request.files["file"]
    if audio_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    musetalk_url = request.form.get("musetalkUrl", "").strip()
    try:
        fps = int(request.form.get("fps", 25))
        batch_size = int(request.form.get("batch_size", 8))
    except (ValueError, TypeError):
        fps = 25
        batch_size = 8

    if not musetalk_url and not MUSETALK_URL:
        return jsonify({"error": "MuseTalk URL is required (form musetalkUrl or env MUSETALK_URL)"}), 400
    if not musetalk_url:
        musetalk_url = MUSETALK_URL

    _clear_dir(ANSWERS_DIR)
    mp3_path = ANSWERS_DIR / "Answer.mp3"

    # Get file extension and mime type
    file_ext = audio_file.filename.split(".")[-1].lower() if "." in audio_file.filename else ""
    mime_type = audio_file.content_type or ""
    
    # Save or convert to MP3
    is_mp3_upload = (mime_type.lower() == "audio/mpeg" or audio_file.filename.lower().endswith(".mp3"))
    if is_mp3_upload:
        audio_file.save(mp3_path)
    else:
        if not PYDUB_AVAILABLE:
            return jsonify({
                "error": "Conversion to MP3 requires pydub. Please install pydub and ffmpeg.",
                "hint": "pip install pydub and ensure ffmpeg is in PATH"
            }), 500
        temp_input = ANSWERS_DIR / f"temp_input.{file_ext or 'bin'}"
        audio_file.save(temp_input)
        try:
            seg = AudioSegment.from_file(temp_input)
            seg = seg.set_channels(1).set_frame_rate(24000)
            seg.export(mp3_path, format="mp3")
        except Exception as e:
            return jsonify({
                "error": f"Failed to convert to MP3: {e}",
                "hint": "Ensure ffmpeg is installed and available in PATH"
            }), 500
        finally:
            try:
                temp_input.unlink(missing_ok=True)
            except TypeError:
                if temp_input.exists():
                    temp_input.unlink()

    file_size = mp3_path.stat().st_size if mp3_path.exists() else 0

    # Forward the audio file directly to MuseTalk server
    forward_info = None
    try:
        forward_info = _forward_answer_and_trigger_inference(
            mp3_path, 
            musetalk_url,
            fps,
            batch_size
        )
    except Exception as e:
        forward_info = {"ok": False, "error": str(e)}

    return jsonify({
        "status": "saved",
        "uploaded_audio": {
            "filename": "Answer.mp3",
            "original_filename": audio_file.filename,
            "file_path": str(mp3_path),
            "file_url": "/answers/Answer.mp3",
            "mime_type": "audio/mpeg",
            "size_bytes": file_size,
        },
        "forwarded_to_musetalk": forward_info,
        "note": "Audio file uploaded and sent directly to MuseTalk (no omni inference)"
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


@app.route("/clear_buffer", methods=["GET"])
def clear_buffer_handler():
    """Clear frame buffer and reset flags."""
    global frame_buffer, processing_complete, start_signal_received
    frame_buffer.clear()
    processing_complete = False
    start_signal_received = False
    return jsonify({'success': True})


@app.route("/get_frame_buffer", methods=["GET"])
def get_frame_buffer_handler():
    """Return frames; supports incremental fetch via ?from_index=N"""
    try:
        from_index_q = request.args.get('from_index')
        if from_index_q is not None:
            try:
                start = max(0, int(from_index_q))
            except ValueError:
                start = 0
            frames_slice = frame_buffer[start:]
            next_index = len(frame_buffer)
        else:
            frames_slice = frame_buffer
            next_index = len(frame_buffer)
        return jsonify({
            'frames': frames_slice,
            'buffer_size': len(frame_buffer),
            'next_index': next_index,
            'processing_complete': processing_complete,
            'start_signal_received': start_signal_received,
        })
    except Exception as e:
        print(f"[Mini-Omni] Error in get_frame_buffer: {e}")
        return jsonify({'error': str(e)}), 500


@app.route("/mjpeg_stream", methods=["GET"])
def mjpeg_stream_handler():
    """Serve MJPEG composed from buffered JPEG frames (base64)."""
    from flask import Response
    boundary = b'--frame\r\n'

    def generate():
        read_index = 0
        first_written = False
        # Wait for at least one frame or completion
        while read_index >= len(frame_buffer) and not processing_complete:
            time.sleep(0.02)
        while True:
            if read_index < len(frame_buffer):
                entry = frame_buffer[read_index]
                read_index += 1
                try:
                    frame_bytes = base64.b64decode(entry['frame_data'])
                except Exception:
                    continue
                yield boundary
                yield b'Content-Type: image/jpeg\r\n\r\n'
                yield frame_bytes
                yield b'\r\n'
                if not first_written:
                    print('[Mini-Omni] MJPEG: first frame written to client')
                    first_written = True
            else:
                if processing_complete and read_index >= len(frame_buffer):
                    print('[Mini-Omni] MJPEG: finished and all buffered frames flushed')
                    break
                time.sleep(0.01)
        yield b'--frame--\r\n'

    headers = {
        'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
    }
    return Response(generate(), status=200, headers=headers)


@app.route("/config", methods=["GET"])
def config_handler():
    return jsonify({'musetalk_url': MUSETALK_URL})


@app.route("/probe_musetalk", methods=["POST"])
def probe_musetalk_handler():
    """Build health URL with stream_base and attempt HTTP GET with basic diagnostics."""
    try:
        musetalk_base_url = MUSETALK_URL
        musetalk_base_url = str(musetalk_base_url).strip()
        if musetalk_base_url.endswith('/'):
            musetalk_base_url = musetalk_base_url[:-1]
        if not (musetalk_base_url.startswith('http://') or musetalk_base_url.startswith('https://')):
            musetalk_base_url = 'http://' + musetalk_base_url

        xf_proto = request.headers.get('X-Forwarded-Proto')
        xf_host = request.headers.get('X-Forwarded-Host')
        scheme = xf_proto or request.scheme
        host = xf_host or request.host
        public_base = f"{scheme}://{host}"
        url = musetalk_base_url + '/health' + f"?stream_base={public_base}"

        # DNS and TCP diagnostics
        from urllib.parse import urlparse as _urlparse
        parsed = _urlparse(url)
        h = parsed.hostname
        p = parsed.port or (443 if parsed.scheme == 'https' else 80)
        resolved_ips = []
        try:
            infos = socket.getaddrinfo(h, p, proto=socket.IPPROTO_TCP)
            for family, _, _, _, sockaddr in infos:
                ip = sockaddr[0]
                if ip not in resolved_ips:
                    resolved_ips.append(ip)
        except Exception as e:
            print(f"[Mini-Omni] DNS resolution failed for {h}: {e}")

        tcp_ok = False
        tcp_error = None
        try:
            with socket.create_connection((h, p), timeout=2):
                tcp_ok = True
        except Exception as e:
            tcp_error = str(e)

        # HTTP GET
        try:
            resp = requests.get(url, headers={'User-Agent': 'AvatarPageProbe/1.0'}, timeout=5)
            ct = resp.headers.get('Content-Type', '')
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            return jsonify({
                'success': resp.status_code == 200,
                'status': resp.status_code,
                'url': url,
                'resolved_ips': resolved_ips,
                'tcp_connect_ok': tcp_ok,
                'tcp_error': tcp_error,
                'content_type': ct,
                'body': body,
            }), 200 if resp.status_code == 200 else 502
        except Exception as e:
            return jsonify({
                'success': False,
                'status': None,
                'url': url,
                'resolved_ips': resolved_ips,
                'tcp_connect_ok': tcp_ok,
                'tcp_error': tcp_error,
                'error': str(e),
            }), 504
    except Exception as e:
        print(f"[Mini-Omni] probe_musetalk error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route("/uploads/<path:filename>")
def serve_upload(filename: str):
    target = UPLOADS_DIR / filename
    if not target.exists() or not target.is_file():
        abort(404)
    resp = send_from_directory(UPLOADS_DIR, filename)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route("/stream_frames", methods=["POST"])
def stream_frames():
    """Receive NDJSON lines with frames from MuseTalk."""
    global frame_buffer, processing_complete, start_signal_received
    try:
        buf = b''
        total_lines = 0
        total_frames_received = 0

        def process_line(line: str):
            nonlocal total_lines, total_frames_received
            total_lines += 1
            line = line.strip()
            if not line:
                return
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.warning('Non-JSON line received; ignoring')
                return

            status = msg.get('status')
            if status == 'start':
                start_signal_received = True
                logger.info('Start signal received')
                return
            if status == 'finished':
                processing_complete = True
                logger.info('Finished signal received')
                return

            frames = msg.get('frames', [])
            # Normalize frames to a list for robustness
            if frames is None:
                frames_list = []
            elif isinstance(frames, dict):
                frames_list = [frames]
            elif isinstance(frames, list):
                frames_list = frames
            else:
                frames_list = []
            if frames_list:
                added = 0
                last_num = None
                for fr in frames_list:
                    b64 = fr.get('frame_data') or fr.get('frame')
                    if not b64:
                        continue
                    last_num = fr.get('frame_number', 0)
                    frame_buffer.append({
                        'frame_number': last_num,
                        'frame_data': b64,
                        'timestamp': datetime.datetime.now().isoformat(),
                    })
                    added += 1
                total_frames_received += added
                logger.info(f"Received {added} frames (last #{last_num}); buffer size={len(frame_buffer)}; total_frames_received={total_frames_received}")

        # Read request data
        chunk = request.get_data(cache=False, as_text=False, parse_form_data=False)
        if chunk:
            buf += chunk
            while True:
                idx = buf.find(b"\n")
                if idx == -1:
                    break
                line_bytes = buf[:idx]
                buf = buf[idx+1:]
                process_line(line_bytes.decode('utf-8', errors='ignore'))

        if buf:
            process_line(buf.decode('utf-8', errors='ignore'))

        return jsonify({'ok': True, 'lines': total_lines, 'frames': total_frames_received}), 200
    except Exception as e:
        logger.exception('stream_frames error')
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
