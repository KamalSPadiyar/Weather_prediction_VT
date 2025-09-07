#!/usr/bin/env python3
"""
Automated Deployment Helper
Creates GitHub repository files and guides through cloud deployment
"""

import os
import shutil
import subprocess
import json
from pathlib import Path

def create_github_ready_package():
    """Create a GitHub-ready package for deployment"""
    
    print("🚀 CREATING GITHUB DEPLOYMENT PACKAGE")
    print("=" * 50)
    
    # Create GitHub deployment directory
    github_dir = Path("github_deploy")
    if github_dir.exists():
        shutil.rmtree(github_dir)
    github_dir.mkdir()
    
    # Essential files for web deployment
    essential_files = [
        ("web_app.py", "Main web application"),
        ("interactive_weather.py", "AI prediction logic"),
        ("main.py", "Model core functions"),
        ("best_multimodal_weather_model.pth", "Trained AI model"),
        ("requirements.txt", "Python dependencies"),
        ("Procfile", "Deployment process file"),
        ("runtime.txt", "Python version specification")
    ]
    
    print("📁 Copying essential files...")
    for file, description in essential_files:
        if os.path.exists(file):
            shutil.copy2(file, github_dir)
            print(f"✅ {file} - {description}")
        else:
            print(f"⚠️  {file} - Not found")
    
    # Copy templates directory
    templates_src = Path("templates")
    templates_dst = github_dir / "templates"
    if templates_src.exists():
        shutil.copytree(templates_src, templates_dst)
        print("✅ templates/ - Web interface files")
    else:
        print("⚠️  templates/ - Not found")
    
    # Create deployment README
    readme_content = """# 🌤️ AI Weather Prediction System

![Weather Prediction](https://img.shields.io/badge/AI-Weather%20Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.10-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)

## 🎯 Live Demo
**Deployed at**: [Your Live Link Here]

## 📱 Features
- 🌍 Global weather prediction for any location
- 🤖 AI-powered tomorrow's temperature forecasts
- 📊 Confidence assessment (High/Medium/Low)
- 🌡️ Temperature trend analysis
- ⚡ Real-time weather data integration
- 🎨 Beautiful responsive web interface

## 🚀 Quick Start

### Local Development
```bash
pip install -r requirements.txt
python web_app.py
```
Visit: http://localhost:5000

### Cloud Deployment
This app is ready to deploy on:
- Render.com (recommended)
- Railway.app
- Heroku
- PythonAnywhere

## 🛠️ Technology Stack
- **Backend**: Python, Flask, PyTorch
- **AI Model**: Multi-modal weather prediction
- **APIs**: OpenWeatherMap for real-time data
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Docker-ready, cloud-optimized

## 📊 System Requirements
- Python 3.7+
- 2GB RAM minimum
- Internet connection
- 50MB storage

## 🌟 How It Works
1. User enters any location worldwide
2. System fetches current weather conditions
3. AI model analyzes weather patterns and trends
4. Provides tomorrow's temperature prediction with confidence level

## 🔧 Configuration
The system uses OpenWeatherMap API. For production use, replace the API key in `web_app.py`.

## 📄 License
MIT License - Feel free to use and modify!

---
*Built with ❤️ using AI and real-time weather data*
"""
    
    with open(github_dir / "README.md", "w", encoding='utf-8') as f:
        f.write(readme_content)
    print("✅ README.md - GitHub documentation")
    
    # Create .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Environment variables
.env
.env.local

# Deployment
dist/
build/
"""
    
    with open(github_dir / ".gitignore", "w") as f:
        f.write(gitignore_content)
    print("✅ .gitignore - Git ignore rules")
    
    # Create app.json for easy deployment
    app_json = {
        "name": "AI Weather Prediction System",
        "description": "AI-powered weather prediction system with web interface",
        "keywords": ["python", "ai", "weather", "prediction", "pytorch", "flask"],
        "website": "https://github.com/yourusername/weather-prediction-system",
        "repository": "https://github.com/yourusername/weather-prediction-system",
        "env": {
            "PYTHON_VERSION": {
                "description": "Python version",
                "value": "3.10.0"
            }
        },
        "buildpacks": [
            {
                "url": "heroku/python"
            }
        ]
    }
    
    with open(github_dir / "app.json", "w") as f:
        json.dump(app_json, f, indent=2)
    print("✅ app.json - App configuration")
    
    print(f"\n🎉 GitHub package created in: {github_dir.absolute()}")
    return github_dir

def create_deployment_instructions():
    """Create step-by-step deployment instructions"""
    
    instructions = """
🌐 DEPLOY YOUR AI WEATHER SYSTEM - STEP BY STEP
===============================================

📋 OPTION 1: RENDER.COM (FREE & RECOMMENDED)
--------------------------------------------

Step 1: Upload to GitHub (5 minutes)
1. Go to: https://github.com
2. Sign up/Login
3. Click "New repository"
4. Name: "weather-prediction-system"
5. Make it Public
6. Upload all files from 'github_deploy' folder

Step 2: Deploy on Render.com (10 minutes)
1. Go to: https://render.com
2. Sign up with GitHub account
3. Click "New" → "Web Service"
4. Connect GitHub repository: weather-prediction-system
5. Configure:
   - Name: weather-prediction-system
   - Environment: Python 3
   - Build Command: pip install -r requirements.txt
   - Start Command: python web_app.py
6. Click "Create Web Service"

Step 3: Get Your Live Link! 🎉
Your app will be available at:
https://weather-prediction-system.onrender.com

📋 OPTION 2: RAILWAY.APP (FASTEST)
---------------------------------
1. Go to: https://railway.app
2. Sign up with GitHub
3. Click "Deploy from GitHub repo"
4. Select: weather-prediction-system
5. Your link: https://your-app.railway.app

📋 OPTION 3: ONE-CLICK DEPLOY BUTTONS
------------------------------------
Add these to your GitHub README:

[![Deploy on Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/deploy)

🎯 EXPECTED TIMELINE:
- GitHub upload: 5 minutes
- Deployment: 5-10 minutes
- TOTAL: ~15 minutes to live link!

💡 TIPS:
- Use Render.com for reliable free hosting
- Your app will auto-sleep after 15min inactivity (free tier)
- First load might take 30 seconds (cold start)
- Perfect for sharing and demos!

🚀 READY TO DEPLOY? START WITH GITHUB!
"""
    with open("DEPLOY_INSTRUCTIONS.txt", "w", encoding='utf-8') as f:
        f.write(instructions)
    
    print("✅ Created DEPLOY_INSTRUCTIONS.txt")

def main():
    """Main deployment helper function"""
    
    print("🌤️ WEATHER PREDICTION SYSTEM - DEPLOYMENT HELPER")
    print("=" * 60)
    print("This will prepare your system for cloud deployment!")
    print()
    
    # Create GitHub package
    github_dir = create_github_ready_package()
    
    # Create instructions
    create_deployment_instructions()
    
    print("\n" + "🎯" * 20)
    print("🚀 DEPLOYMENT PACKAGE READY!")
    print("🎯" * 20)
    
    print(f"\n📁 Files ready for GitHub: {github_dir.absolute()}")
    print("\n🌐 TO GET YOUR LIVE WEB LINK:")
    print("1. Upload 'github_deploy' folder contents to GitHub")
    print("2. Deploy on Render.com (free)")
    print("3. Get live link: https://your-app.onrender.com")
    
    print(f"\n📋 Detailed instructions: DEPLOY_INSTRUCTIONS.txt")
    
    print("\n✨ WHAT USERS WILL GET:")
    print("- Beautiful web interface")
    print("- AI weather predictions for any location")
    print("- Real-time weather data")
    print("- Mobile-friendly design")
    print("- No installation required")
    
    print(f"\n🎉 Ready to deploy? Upload '{github_dir}' to GitHub!")

if __name__ == "__main__":
    main()
