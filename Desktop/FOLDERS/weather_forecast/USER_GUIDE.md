# 🌤️ Weather Prediction System - User Guide

## How to Use

### Option 1: Interactive Mode (Recommended)
```bash
python interactive_weather.py
```
- Choose option 1 for interactive mode
- Enter any place name when prompted
- Get detailed weather predictions with confidence levels
- Continue predicting for multiple places

### Option 2: Simple Mode
```bash
python simple_weather.py
```
- Just enter place names one by one
- Quick and easy predictions

### Option 3: Demo Mode
```bash
python interactive_weather.py
```
- Choose option 2 for demo mode
- See predictions for famous cities worldwide

## What It Does

🎯 **Input**: Any place name (e.g., "Mumbai", "New York", "Tokyo")

🧠 **AI Processing**: 
- Downloads real satellite imagery for that location
- Gets current weather conditions from live APIs
- Uses Vision Transformer AI to predict tomorrow's temperature

📊 **Output**: 
- Current weather conditions
- Tomorrow's temperature prediction
- Temperature change trend
- Confidence level

## Examples

### Input Examples:
✅ "Mumbai" → Mumbai, Maharashtra, IN  
✅ "New York" → New York, New York, US  
✅ "London" → London, England, GB  
✅ "Tokyo" → Tokyo, Japan  
✅ "Nainital" → Nainital, Uttarakhand, IN  
✅ "Paris, France" → Paris, France  
✅ "Sydney, Australia" → Sydney, NSW, AU  

### Sample Output:
```
🎯 WEATHER PREDICTION RESULTS
📍 LOCATION: Mumbai, Maharashtra, IN
☁️ CURRENT CONDITIONS:
   🌡️ Temperature: 30.2°C
   💧 Humidity: 73%
   ☁️ Cloud Cover: 90%
🔮 TOMORROW'S AI PREDICTION:
   🌡️ Temperature: 33.2°C
   📈 Change: +3.0°C (🔥 Warmer)
   ✅ Confidence: High
```

## Features

✅ **Global Coverage**: Works for any city worldwide  
✅ **Real Satellite Data**: Downloads actual satellite imagery  
✅ **Live Weather Data**: Current conditions from OpenWeatherMap  
✅ **AI-Powered**: Vision Transformer + Multi-modal fusion  
✅ **User-Friendly**: Simple place name input  
✅ **Detailed Output**: Temperature, humidity, pressure, wind, etc.  
✅ **Trend Analysis**: Shows if getting warmer/cooler  
✅ **Confidence Levels**: High/Medium/Low prediction confidence  

## Tips

💡 **Be Specific**: If location not found, try adding country (e.g., "Mumbai, India")  
💡 **Try Variations**: "NYC" vs "New York" vs "New York City"  
💡 **Multiple Predictions**: You can predict for multiple cities in one session  
💡 **Exit Anytime**: Type 'exit' or 'quit' to stop  

## Accuracy

🎯 **Model Accuracy**: 93.54% R² score  
🎯 **Temperature Error**: ±2-3°C average  
🎯 **Best For**: Next day temperature predictions  

## Requirements

✅ Internet connection (for satellite data and weather APIs)  
✅ Python with required packages installed  
✅ Trained AI model (run `python main.py` first)  

Enjoy predicting the weather! 🌤️
