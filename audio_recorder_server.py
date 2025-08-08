import os
import base64
import json
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Create recordings directory if it doesn't exist
RECORDINGS_DIR = 'recordings'
if not os.path.exists(RECORDINGS_DIR):
    os.makedirs(RECORDINGS_DIR)

@app.route('/')
def index():
    """Serve the main recording page"""
    return render_template('audio_recorder.html')

@app.route('/save_audio', methods=['POST'])
def save_audio():
    """Save the recorded audio to the recordings folder"""
    try:
        data = request.get_json()
        audio_data = data.get('audio_data')
        
        if not audio_data:
            return jsonify({'error': 'No audio data received'}), 400
        
        # Remove the data URL prefix to get just the base64 data
        if audio_data.startswith('data:audio/wav;base64,'):
            audio_data = audio_data.split(',')[1]
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'recording_{timestamp}.wav'
        filepath = os.path.join(RECORDINGS_DIR, filename)
        
        # Save the audio file
        with open(filepath, 'wb') as f:
            f.write(audio_bytes)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': f'Audio saved as {filename}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recordings')
def list_recordings():
    """List all recorded audio files"""
    try:
        files = []
        for filename in os.listdir(RECORDINGS_DIR):
            if filename.endswith('.wav'):
                filepath = os.path.join(RECORDINGS_DIR, filename)
                file_size = os.path.getsize(filepath)
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                files.append({
                    'filename': filename,
                    'size': file_size,
                    'created': file_time.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({'recordings': files})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recordings/<filename>')
def get_recording(filename):
    """Serve a specific recording file"""
    try:
        filepath = os.path.join(RECORDINGS_DIR, filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='audio/wav')
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recordings/<filename>', methods=['DELETE'])
def delete_recording(filename):
    """Delete a specific recording file"""
    try:
        filepath = os.path.join(RECORDINGS_DIR, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True, 'message': f'Deleted {filename}'})
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Audio Recorder Server starting...")
    print(f"Recordings will be saved to: {os.path.abspath(RECORDINGS_DIR)}")
    print(f"Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=True)
