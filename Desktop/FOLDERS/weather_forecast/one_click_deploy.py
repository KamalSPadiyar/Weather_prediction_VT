#!/usr/bin/env python3
"""
ONE-CLICK DEPLOY SCRIPT
Get your live web link in under 5 minutes!
"""

import os
import subprocess
import webbrowser
import json
from pathlib import Path

def deploy_to_railway():
    """Deploy directly to Railway.app for instant live link"""
    
    print("🚀 ONE-CLICK DEPLOY TO RAILWAY.APP")
    print("=" * 40)
    print("This will give you a live web link in under 5 minutes!")
    print()
    
    # Check if Railway CLI is installed
    try:
        result = subprocess.run(['railway', '--version'], capture_output=True, text=True)
        print("✅ Railway CLI detected")
    except FileNotFoundError:
        print("📦 Installing Railway CLI...")
        print("Visit: https://railway.app/cli")
        print("Or run: npm install -g @railway/cli")
        return False
    
    # Change to deployment directory
    os.chdir(Path(__file__).parent / "github_deploy")
    
    print("🔐 Logging into Railway...")
    subprocess.run(['railway', 'login'])
    
    print("🚀 Creating new Railway project...")
    subprocess.run(['railway', 'init'])
    
    print("📦 Deploying to Railway...")
    subprocess.run(['railway', 'up'])
    
    print("🌐 Getting your live web link...")
    result = subprocess.run(['railway', 'domain'], capture_output=True, text=True)
    
    if result.stdout:
        live_url = result.stdout.strip()
        print(f"🎉 SUCCESS! Your live web link is:")
        print(f"🔗 {live_url}")
        print()
        print("✨ Features available:")
        print("- AI weather predictions for any location")
        print("- Beautiful responsive web interface")
        print("- Real-time weather data")
        print("- Mobile-friendly design")
        print()
        
        # Open in browser
        webbrowser.open(live_url)
        return live_url
    else:
        print("⚠️ Could not retrieve domain. Check Railway dashboard.")
        return None

def deploy_manual_guide():
    """Show manual deployment guide for alternative methods"""
    
    print("📋 MANUAL DEPLOYMENT OPTIONS")
    print("=" * 30)
    print()
    print("🌟 OPTION 1: Render.com (Recommended)")
    print("1. Upload 'github_deploy' to GitHub")
    print("2. Connect GitHub to Render.com")
    print("3. Deploy → Get live link")
    print("Time: ~10 minutes")
    print()
    print("🌟 OPTION 2: Railway.app")
    print("1. Install Railway CLI: npm install -g @railway/cli")
    print("2. Run this script again")
    print("Time: ~5 minutes")
    print()
    print("🌟 OPTION 3: Heroku")
    print("1. Install Heroku CLI")
    print("2. heroku create your-app-name")
    print("3. git push heroku main")
    print()
    
    # Open deployment guides
    print("📖 Opening deployment guides...")
    webbrowser.open("https://render.com")
    webbrowser.open("https://railway.app")

def main():
    """Main deployment function"""
    
    print("🌤️ AI WEATHER PREDICTION SYSTEM")
    print("ONE-CLICK DEPLOYMENT")
    print("=" * 50)
    print()
    
    # Check if we're in the right directory
    if not Path("github_deploy").exists():
        print("❌ Error: github_deploy folder not found")
        print("Please run this from the weather_forecast directory")
        return
    
    print("Choose deployment option:")
    print("1. 🚀 One-click Railway deploy (fastest)")
    print("2. 📋 Manual deployment guide")
    print("3. 🌐 Open cloud provider websites")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        live_url = deploy_to_railway()
        if live_url:
            print(f"\n🎉 DEPLOYMENT COMPLETE!")
            print(f"🔗 Live web link: {live_url}")
            
            # Save the live link
            with open("LIVE_LINK.txt", "w") as f:
                f.write(f"🌤️ AI Weather Prediction System\n")
                f.write(f"Live Web Link: {live_url}\n")
                f.write(f"Deployment: Railway.app\n")
                f.write(f"Status: Active\n")
    
    elif choice == "2":
        deploy_manual_guide()
    
    elif choice == "3":
        print("🌐 Opening cloud deployment websites...")
        webbrowser.open("https://render.com")
        webbrowser.open("https://railway.app")
        webbrowser.open("https://heroku.com")
        webbrowser.open("https://pythonanywhere.com")
    
    else:
        print("Invalid choice. Please run again.")

if __name__ == "__main__":
    main()
