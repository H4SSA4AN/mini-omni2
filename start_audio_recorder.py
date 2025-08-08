#!/usr/bin/env python3
"""
Startup script for the WebRTC Audio Recorder
Checks dependencies and starts the server
"""

import sys
import subprocess
import importlib.util

def check_dependency(module_name, package_name=None):
    """Check if a Python module is available"""
    if package_name is None:
        package_name = module_name
    
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ {package_name} is not installed")
        return False
    else:
        print(f"✅ {package_name} is available")
        return True

def install_dependency(package_name):
    """Install a Python package using pip"""
    try:
        print(f"📦 Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def check_dependencies():
    """Check and install required dependencies"""
    print("🔍 Checking dependencies...")
    print("-" * 30)
    
    dependencies = [
        ("flask", "Flask"),
        ("flask_cors", "Flask-CORS")
    ]
    
    missing_deps = []
    for module_name, package_name in dependencies:
        if not check_dependency(module_name, package_name):
            missing_deps.append(package_name)
    
    if missing_deps:
        print()
        print("📦 Installing missing dependencies...")
        for package in missing_deps:
            if not install_dependency(package):
                print(f"❌ Failed to install {package}. Please install manually:")
                print(f"   pip install {package}")
                return False
    
    print()
    print("✅ All dependencies are ready!")
    return True

def start_server():
    """Start the audio recorder server"""
    print("🚀 Starting WebRTC Audio Recorder Server...")
    print("=" * 50)
    
    try:
        # Import and run the server
        from audio_recorder_server import app
        
        print("✅ Server started successfully!")
        print("🌐 Open http://localhost:5000 in your browser")
        print("📁 Recordings will be saved to the 'recordings' folder")
        print("⏹️  Press Ctrl+C to stop the server")
        print()
        
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except ImportError as e:
        print(f"❌ Error importing server: {e}")
        return False
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def main():
    print("🎤 WebRTC Audio Recorder")
    print("=" * 30)
    print()
    
    # Check dependencies
    if not check_dependencies():
        print()
        print("❌ Dependency check failed. Please install missing packages manually.")
        return
    
    print()
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
