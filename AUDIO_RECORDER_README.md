# WebRTC Audio Recorder

A modern web application for recording audio using WebRTC technology, built with Python Flask and JavaScript.

## Features

- 🎤 **Real-time Audio Recording**: Record audio directly in your browser using WebRTC
- 🎨 **Modern UI**: Beautiful, responsive interface with audio visualization
- 💾 **File Management**: Save, play, download, and delete recordings
- 📁 **Organized Storage**: All recordings are saved to a `recordings` folder with timestamps
- 🔄 **Live Visualization**: Real-time audio spectrum analyzer during recording
- 📱 **Mobile Friendly**: Works on desktop and mobile devices

## Installation

1. Make sure you have the required dependencies:
```bash
pip install -r requirements.txt
```

2. The application will automatically create a `recordings` folder when started.

## Usage

1. **Start the server**:
```bash
cd mini-omni2
python audio_recorder_server.py
```

2. **Open your browser** and navigate to:
```
http://localhost:5000
```

3. **Grant microphone permissions** when prompted by your browser.

4. **Record audio**:
   - Click the red "Record" button to start recording
   - Click "Stop" when finished
   - Click "Save" to save the recording to the server

5. **Manage recordings**:
   - View all saved recordings in the list
   - Play recordings directly in the browser
   - Download recordings to your computer
   - Delete unwanted recordings

## File Structure

```
mini-omni2/
├── audio_recorder_server.py    # Flask server for audio recording
├── templates/
│   └── audio_recorder.html     # Web interface
├── recordings/                 # Saved audio files (created automatically)
└── requirements.txt           # Python dependencies
```

## Technical Details

- **Backend**: Python Flask server
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Audio Format**: WAV (44.1kHz, 16-bit)
- **WebRTC**: Uses MediaRecorder API for browser-based recording
- **Visualization**: Canvas-based audio spectrum analyzer

## API Endpoints

- `GET /` - Main recording interface
- `POST /save_audio` - Save recorded audio
- `GET /recordings` - List all recordings
- `GET /recordings/<filename>` - Download a specific recording
- `DELETE /recordings/<filename>` - Delete a specific recording

## Browser Compatibility

- Chrome 47+
- Firefox 44+
- Safari 14+
- Edge 79+

## Security Notes

- The application requires microphone access
- All audio processing happens locally in the browser
- Files are saved on the server in the `recordings` folder
- No audio data is transmitted to third parties

## Troubleshooting

1. **Microphone not working**: Make sure to allow microphone access in your browser
2. **Recording not saving**: Check that the `recordings` folder has write permissions
3. **Audio quality issues**: Try using a different microphone or check browser settings

## License

This project is part of the OmniServer project and follows the same license terms.
