"""
Deployment Script for Weather Prediction System
Creates a standalone executable and deployment package
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def create_deployment_package():
    """Create a complete deployment package"""
    
    print("🚀 WEATHER PREDICTION SYSTEM DEPLOYMENT")
    print("="*50)
    
    # Create deployment directory
    deploy_dir = Path("weather_forecast_deploy")
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()
    
    print("📁 Creating deployment package...")
    
    # Copy essential files
    essential_files = [
        "interactive_weather.py",
        "main.py",
        "requirements.txt",
        "best_multimodal_weather_model.pth"
    ]
    
    for file in essential_files:
        if os.path.exists(file):
            shutil.copy2(file, deploy_dir)
            print(f"✅ Copied {file}")
        else:
            print(f"⚠️ Warning: {file} not found")
    
    # Create run script for Windows
    run_script_content = """@echo off
echo Starting Weather Prediction System...
echo.
python interactive_weather.py
pause
"""
    
    with open(deploy_dir / "run_weather_system.bat", "w") as f:
        f.write(run_script_content)
    print("✅ Created run_weather_system.bat")
    
    # Create run script for Linux/Mac
    run_script_unix = """#!/bin/bash
echo "Starting Weather Prediction System..."
echo
python3 interactive_weather.py
read -p "Press Enter to exit..."
"""
    
    with open(deploy_dir / "run_weather_system.sh", "w") as f:
        f.write(run_script_unix)
    
    # Make it executable on Unix systems
    try:
        os.chmod(deploy_dir / "run_weather_system.sh", 0o755)
        print("✅ Created run_weather_system.sh")
    except:
        print("✅ Created run_weather_system.sh (permissions may need manual setting)")
    
    # Create installation script
    install_script = """@echo off
echo Installing Weather Prediction System Dependencies...
echo.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Installation complete!
echo Run "run_weather_system.bat" to start the application.
pause
"""
    
    with open(deploy_dir / "install_dependencies.bat", "w") as f:
        f.write(install_script)
    print("✅ Created install_dependencies.bat")
    
    # Create Unix installation script
    install_script_unix = """#!/bin/bash
echo "Installing Weather Prediction System Dependencies..."
echo
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
echo
echo "Installation complete!"
echo "Run './run_weather_system.sh' to start the application."
"""
    
    with open(deploy_dir / "install_dependencies.sh", "w") as f:
        f.write(install_script_unix)
    
    try:
        os.chmod(deploy_dir / "install_dependencies.sh", 0o755)
        print("✅ Created install_dependencies.sh")
    except:
        print("✅ Created install_dependencies.sh (permissions may need manual setting)")
      # Create README for deployment
    readme_content = """# Weather Prediction System

## Quick Start

### Windows:
1. Run `install_dependencies.bat` (first time only)
2. Run `run_weather_system.bat` to start the application

### Linux/Mac:
1. Run `./install_dependencies.sh` (first time only)
2. Run `./run_weather_system.sh` to start the application

## Manual Installation

If the automatic scripts don't work:

```bash
pip install -r requirements.txt
python interactive_weather.py
```

## Features

- AI-powered weather prediction
- Real-time weather data from OpenWeatherMap
- Support for any location worldwide
- Temperature trend analysis
- Confidence assessment
- Interactive command-line interface

## System Requirements

- Python 3.7 or higher
- Internet connection
- 2GB RAM minimum
- Works on Windows, Linux, and macOS

## Usage

1. Enter any place name (e.g., "New York", "London", "Mumbai, India")
2. Get current weather conditions
3. Receive AI prediction for tomorrow's temperature
4. View confidence level and temperature trend

## Support

If you encounter any issues:
1. Ensure Python is properly installed
2. Check your internet connection
3. Verify all dependencies are installed
4. Make sure the model file (best_multimodal_weather_model.pth) is present

## API Key

The system uses OpenWeatherMap API. The included key has usage limits.
For heavy usage, get your own free API key from: https://openweathermap.org/api
"""
    
    with open(deploy_dir / "README.md", "w", encoding='utf-8') as f:
        f.write(readme_content)
    print("✅ Created README.md")
      # Create version info
    version_info = """Weather Prediction System v1.0
Built: June 22, 2025
Author: AI Weather Prediction Team

Components:
- Interactive Weather Prediction System
- Multi-modal AI Model (CPU Optimized)
- OpenWeatherMap Integration
- Cross-platform Support

License: MIT
"""
    
    with open(deploy_dir / "VERSION.txt", "w", encoding='utf-8') as f:
        f.write(version_info)
    print("✅ Created VERSION.txt")
    
    print(f"\n🎉 Deployment package created in: {deploy_dir.absolute()}")
    print("\n📦 Package contents:")
    for item in deploy_dir.iterdir():
        size = item.stat().st_size if item.is_file() else "DIR"
        print(f"   {item.name} ({size} bytes)" if isinstance(size, int) else f"   {item.name} ({size})")
    
    print(f"\n✅ Ready for distribution!")
    print(f"📁 Share the entire '{deploy_dir}' folder")
    
    return deploy_dir

def create_portable_version():
    """Create a more portable version with embedded dependencies"""
    
    print("\n🔧 Creating portable version...")
    
    try:
        # Check if PyInstaller is available
        subprocess.run([sys.executable, "-c", "import PyInstaller"], 
                      check=True, capture_output=True)
        
        print("📦 Creating standalone executable with PyInstaller...")
        
        # PyInstaller command
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--console",
            "--name", "WeatherPredictor",
            "--add-data", "best_multimodal_weather_model.pth;.",
            "interactive_weather.py"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Standalone executable created in 'dist' folder")
            
            # Move executable to deployment folder
            exe_name = "WeatherPredictor.exe" if os.name == 'nt' else "WeatherPredictor"
            if os.path.exists(f"dist/{exe_name}"):
                deploy_dir = Path("weather_forecast_deploy")
                shutil.copy2(f"dist/{exe_name}", deploy_dir)
                print(f"✅ Copied {exe_name} to deployment package")
        else:
            print("⚠️ PyInstaller failed, skipping executable creation")
            print(f"Error: {result.stderr}")
            
    except subprocess.CalledProcessError:
        print("⚠️ PyInstaller not available, skipping executable creation")
        print("💡 Install with: pip install pyinstaller")

def main():
    """Main deployment function"""
    
    # Check if we're in the right directory
    if not os.path.exists("interactive_weather.py"):
        print("❌ Error: interactive_weather.py not found!")
        print("Please run this script from the weather_forecast directory")
        return
    
    # Create deployment package
    deploy_dir = create_deployment_package()
    
    # Try to create portable version
    create_portable_version()
    
    print("\n" + "🌤️ " * 20)
    print("🚀 DEPLOYMENT COMPLETE!")
    print("🌤️ " * 20)
    
    print(f"\n📦 Distribution package: {deploy_dir.absolute()}")
    print("\n🎯 Next steps:")
    print("1. Test the deployment by running install_dependencies.bat/sh")
    print("2. Test the application with run_weather_system.bat/sh")
    print("3. Share the entire deployment folder")
    print("4. Recipients should run install script first, then run script")
    
    print("\n💡 Deployment Tips:")
    print("- The package works on Windows, Linux, and macOS")
    print("- Users need Python 3.7+ installed")
    print("- Internet connection required for weather data")
    print("- All dependencies will be automatically installed")

if __name__ == "__main__":
    main()
