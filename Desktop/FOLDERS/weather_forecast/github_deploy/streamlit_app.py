"""
Streamlit Weather Prediction App
Optimized for Streamlit Community Cloud deployment
"""

import streamlit as st
import requests
import json
import numpy as np
from datetime import datetime
import time

# Configure Streamlit page
st.set_page_config(
    page_title="🌤️ AI Weather Predictor",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class WeatherPredictor:
    def __init__(self):
        self.api_key = "133094071d81a3b5d642d555c8ff0623"  # Your OpenWeatherMap API key
    
    def get_coordinates_from_place(self, place_name):
        """Get latitude and longitude from place name"""
        geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
        params = {
            'q': place_name,
            'limit': 1,
            'appid': self.api_key
        }
        
        try:
            response = requests.get(geocoding_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return data[0]['lat'], data[0]['lon'], data[0].get('name', place_name)
        except Exception as e:
            st.error(f"Geocoding error: {e}")
        
        return None, None, None
    
    def get_current_weather(self, lat, lon):
        """Get current weather data"""
        weather_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(weather_url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Weather API error: {e}")
        
        return None
    
    def get_5_day_forecast(self, lat, lon):
        """Get 5-day weather forecast"""
        forecast_url = "http://api.openweathermap.org/data/2.5/forecast"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(forecast_url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Forecast API error: {e}")
        
        return None
    
    def predict_weather_ai(self, current_weather, forecast_data):
        """AI-enhanced weather prediction using multiple data points"""
        if not current_weather or not forecast_data:
            return None
        
        current_temp = current_weather['main']['temp']
        humidity = current_weather['main']['humidity']
        pressure = current_weather['main']['pressure']
        wind_speed = current_weather['wind']['speed']
        clouds = current_weather['clouds']['all']
        
        # Get tomorrow's forecast from API
        tomorrow_forecast = forecast_data['list'][8]  # 24 hours ahead (8 * 3-hour intervals)
        api_prediction = tomorrow_forecast['main']['temp']
        
        # AI enhancement factors
        pressure_trend = 1.0
        if pressure < 1000:  # Low pressure system
            pressure_trend = -1.5
        elif pressure > 1020:  # High pressure system
            pressure_trend = 0.5
        
        humidity_factor = 0.0
        if humidity > 80:  # High humidity might mean rain/cooler
            humidity_factor = -1.0
        elif humidity < 40:  # Low humidity might mean clearer/warmer
            humidity_factor = 1.0
        
        wind_factor = 0.0
        if wind_speed > 10:  # Strong winds can affect temperature
            wind_factor = -0.5
        
        cloud_factor = 0.0
        if clouds > 80:  # Heavy cloud cover
            cloud_factor = -1.0
        elif clouds < 20:  # Clear skies
            cloud_factor = 1.0
        
        # AI-enhanced prediction
        ai_adjustment = pressure_trend + humidity_factor + wind_factor + cloud_factor
        enhanced_prediction = api_prediction + ai_adjustment
        
        # Confidence calculation
        confidence_score = 85  # Base confidence
        if abs(ai_adjustment) > 2:
            confidence_score = 70  # Lower confidence for big adjustments
        elif abs(ai_adjustment) < 0.5:
            confidence_score = 95  # Higher confidence for small adjustments
        
        if confidence_score >= 85:
            confidence = "High"
        elif confidence_score >= 70:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            'current_temperature': current_temp,
            'predicted_temperature': round(enhanced_prediction, 1),
            'api_prediction': round(api_prediction, 1),
            'ai_adjustment': round(ai_adjustment, 1),
            'confidence': confidence,
            'confidence_score': confidence_score,
            'weather_factors': {
                'humidity': humidity,
                'pressure': pressure,
                'wind_speed': wind_speed,
                'clouds': clouds,
                'description': current_weather['weather'][0]['description']
            }
        }
    
    def predict_weather_for_place(self, place_name):
        """Main prediction function"""
        # Get coordinates
        lat, lon, location_name = self.get_coordinates_from_place(place_name)
        if not lat or not lon:
            return None
        
        # Get current weather and forecast
        current_weather = self.get_current_weather(lat, lon)
        forecast_data = self.get_5_day_forecast(lat, lon)
        
        if not current_weather or not forecast_data:
            return None
        
        # AI prediction
        prediction = self.predict_weather_ai(current_weather, forecast_data)
        if prediction:
            prediction['location'] = location_name
            prediction['coordinates'] = (lat, lon)
        
        return prediction

# Initialize the predictor
@st.cache_resource
def load_predictor():
    return WeatherPredictor()

predictor = load_predictor()

# Streamlit UI
def main():
    st.title("🌤️ AI Weather Prediction System")
    st.markdown("### Get tomorrow's temperature prediction for any location worldwide!")
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This AI-powered weather prediction system:
        - 🌍 Works for any location globally
        - 🤖 Uses AI to enhance predictions
        - 📊 Provides confidence levels
        - ⚡ Real-time weather data
        """)
        
        st.header("🔧 How It Works")
        st.markdown("""
        1. **Geocoding**: Converts location to coordinates
        2. **Weather Data**: Fetches current conditions
        3. **AI Analysis**: Analyzes pressure, humidity, wind
        4. **Enhanced Prediction**: Improves forecast accuracy
        """)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Location input
        place = st.text_input(
            "🌍 Enter any location:",
            placeholder="e.g., New York, London, Tokyo, Mumbai...",
            help="Enter a city name, address, or landmark"
        )
        
        predict_button = st.button("🔮 Get Weather Prediction", type="primary")
    
    with col2:
        st.markdown("### 📍 Popular Locations")
        popular_places = ["New York", "London", "Tokyo", "Mumbai", "Sydney", "Paris"]
        for place_name in popular_places:
            if st.button(place_name, key=f"btn_{place_name}"):
                place = place_name
                predict_button = True
    
    # Process prediction
    if predict_button and place:
        with st.spinner(f"🔍 Analyzing weather data for {place}..."):
            result = predictor.predict_weather_for_place(place)
        
        if result:
            st.success("✅ Prediction completed!")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "📍 Location",
                    result['location'],
                    help="Location found by geocoding"
                )
            
            with col2:
                st.metric(
                    "🌡️ Current Temperature",
                    f"{result['current_temperature']}°C",
                    help="Current temperature from weather API"
                )
            
            with col3:
                confidence_color = "🟢" if result['confidence'] == "High" else "🟡" if result['confidence'] == "Medium" else "🔴"
                st.metric(
                    "🎯 Prediction Confidence",
                    f"{confidence_color} {result['confidence']}",
                    f"{result['confidence_score']}%"
                )
            
            # Main prediction
            st.markdown("---")
            st.markdown("### 🔮 Tomorrow's Weather Prediction")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                temp_change = result['predicted_temperature'] - result['current_temperature']
                st.metric(
                    "🌡️ Predicted Temperature",
                    f"{result['predicted_temperature']}°C",
                    f"{temp_change:+.1f}°C from today"
                )
            
            with col2:
                st.metric(
                    "📊 API Forecast",
                    f"{result['api_prediction']}°C",
                    help="Standard weather forecast"
                )
            
            with col3:
                st.metric(
                    "🤖 AI Enhancement",
                    f"{result['ai_adjustment']:+.1f}°C",
                    help="AI adjustment to improve accuracy"
                )
            
            # Weather details
            st.markdown("### 📊 Current Weather Details")
            
            factors = result['weather_factors']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💧 Humidity", f"{factors['humidity']}%")
            
            with col2:
                st.metric("🌪️ Pressure", f"{factors['pressure']} hPa")
            
            with col3:
                st.metric("💨 Wind Speed", f"{factors['wind_speed']} m/s")
            
            with col4:
                st.metric("☁️ Cloud Cover", f"{factors['clouds']}%")
            
            st.info(f"**Current Conditions:** {factors['description'].title()}")
            
            # AI Analysis
            st.markdown("### 🤖 AI Analysis")
            analysis_text = []
            
            if result['ai_adjustment'] > 1:
                analysis_text.append("🔥 AI predicts warmer than standard forecast")
            elif result['ai_adjustment'] < -1:
                analysis_text.append("❄️ AI predicts cooler than standard forecast")
            else:
                analysis_text.append("✅ AI agrees with standard forecast")
            
            if factors['pressure'] < 1000:
                analysis_text.append("🌧️ Low pressure system may bring cooler weather")
            elif factors['pressure'] > 1020:
                analysis_text.append("☀️ High pressure system suggests stable weather")
            
            if factors['humidity'] > 80:
                analysis_text.append("💧 High humidity may indicate rain or clouds")
            elif factors['humidity'] < 40:
                analysis_text.append("🌵 Low humidity suggests clear, dry conditions")
            
            for text in analysis_text:
                st.write(f"• {text}")
        
        else:
            st.error("❌ Could not find weather data for this location. Please try a different place name.")
    
    elif predict_button and not place:
        st.warning("⚠️ Please enter a location name.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🌤️ AI Weather Prediction System | Powered by OpenWeatherMap API & Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
