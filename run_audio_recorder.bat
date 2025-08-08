@echo off
echo Starting WebRTC Audio Recorder Server...
echo.
echo Make sure you have installed the requirements:
echo pip install -r requirements.txt
echo.
echo The server will start on http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python audio_recorder_server.py

pause
