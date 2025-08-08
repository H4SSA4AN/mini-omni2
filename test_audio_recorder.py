#!/usr/bin/env python3
"""
Simple test script for the audio recorder server
"""

import requests
import os
import sys

def test_server():
    """Test if the audio recorder server is running"""
    try:
        # Test if server is running
        response = requests.get('http://localhost:5000/', timeout=5)
        if response.status_code == 200:
            print("✅ Server is running successfully!")
            print("🌐 Open http://localhost:5000 in your browser")
            return True
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is it running?")
        print("💡 Start the server with: python audio_recorder_server.py")
        return False
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

def test_recordings_folder():
    """Test if recordings folder exists"""
    recordings_dir = 'recordings'
    if os.path.exists(recordings_dir):
        print(f"✅ Recordings folder exists: {recordings_dir}")
        return True
    else:
        print(f"❌ Recordings folder not found: {recordings_dir}")
        print("💡 The folder will be created when the server starts")
        return False

def main():
    print("🎤 WebRTC Audio Recorder - Server Test")
    print("=" * 40)
    
    # Test recordings folder
    test_recordings_folder()
    print()
    
    # Test server
    if test_server():
        print()
        print("🎉 All tests passed! Your audio recorder is ready to use.")
        print()
        print("📋 Next steps:")
        print("1. Open http://localhost:5000 in your browser")
        print("2. Allow microphone access when prompted")
        print("3. Click 'Record' to start recording")
        print("4. Click 'Stop' when finished")
        print("5. Click 'Save' to save the recording")
    else:
        print()
        print("🔧 Troubleshooting:")
        print("1. Make sure you have installed requirements: pip install -r requirements.txt")
        print("2. Start the server: python audio_recorder_server.py")
        print("3. Check if port 5000 is available")

if __name__ == "__main__":
    main()
