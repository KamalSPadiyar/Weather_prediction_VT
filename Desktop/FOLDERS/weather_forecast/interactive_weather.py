"""
Interactive Weather Prediction System
User enters a place name and gets tomorrow's weather prediction
"""

import torch
import numpy as np
import pandas as pd
from main import MultiModalWeatherPredictor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import requests
import json

class InteractiveWeatherPredictor:
    def __init__(self, model_path='best_multimodal_weather_model.pth'):
        """Initialize the interactive weather predictor"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.api_key = "133094071d81a3b5d642d555c8ff0623"  # OpenWeatherMap API key
        
        # Load the trained model
        try:
            self.model = MultiModalWeatherPredictor(
                img_size=48 if self.device.type == 'cpu' else 64,
                patch_size=6 if self.device.type == 'cpu' else 8,
                weather_features=8
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("✅ AI Weather Model loaded successfully!")
        except:
            print("❌ Model not found. Please train the model first by running: python main.py")
            self.model = None
            return
            
        # Initialize scaler with reasonable weather data ranges
        self.scaler = StandardScaler()
        dummy_data = np.array([
            [15, 60, 1013, 5, 50, 2, 10, 5],    # Average conditions
            [35, 90, 1040, 15, 100, 10, 20, 10], # High values  
            [-5, 20, 980, 0, 0, 0, 1, 0]        # Low values
        ])
        self.scaler.fit(dummy_data)
        
        self.feature_names = ['temperature', 'humidity', 'pressure', 'wind_speed', 
                             'cloud_cover', 'precipitation', 'visibility', 'uv_index']
    
    def get_coordinates_from_place(self, place_name):
        """Get latitude and longitude from place name using OpenWeatherMap Geocoding API"""
        geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
        params = {
            'q': place_name,
            'limit': 1,
            'appid': self.api_key
        }
        
        try:
            response = requests.get(geocoding_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data:
                    location = data[0]
                    return {
                        'lat': location['lat'],
                        'lon': location['lon'],
                        'name': location['name'],
                        'country': location.get('country', ''),
                        'state': location.get('state', '')
                    }
                else:
                    return None
            else:
                print(f"Geocoding API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting coordinates: {e}")
            return None
    
    def get_current_weather_data(self, lat, lon):
        """Get current weather data from OpenWeatherMap API"""
        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(weather_url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                # Extract weather features
                weather_data = {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data.get('wind', {}).get('speed', 0),
                    'cloud_cover': data['clouds']['all'],
                    'precipitation': data.get('rain', {}).get('1h', 0) + data.get('snow', {}).get('1h', 0),
                    'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
                    'uv_index': 5  # Default UV index (would need separate API call)
                }
                
                return weather_data, data
            else:
                print(f"Weather API error: {response.status_code}")
                return None, None
                
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None, None
    
    def predict_weather_for_place(self, place_name):
        """Main function to predict weather for a place name"""
        if self.model is None:
            print("❌ Model not loaded. Please train the model first.")
            return None
        
        print(f"\n🔍 Searching for: {place_name}")
        print("="*60)
        
        # Get coordinates from place name
        location_info = self.get_coordinates_from_place(place_name)
        if not location_info:
            print(f"❌ Could not find location: {place_name}")
            print("💡 Try being more specific (e.g., 'Mumbai, India' instead of 'Mumbai')")
            return None
        
        lat, lon = location_info['lat'], location_info['lon']
        full_location = f"{location_info['name']}"
        if location_info['state']:
            full_location += f", {location_info['state']}"
        if location_info['country']:
            full_location += f", {location_info['country']}"
        
        print(f"📍 Found: {full_location}")
        print(f"🌐 Coordinates: {lat:.4f}°N, {lon:.4f}°E")
        
        # Get current weather data
        print("📡 Fetching current weather conditions...")
        weather_data, raw_weather = self.get_current_weather_data(lat, lon)
        
        if weather_data is None:
            print("❌ Could not fetch weather data for this location")
            return None
        
        # Create a simple synthetic image for the AI model (no display)
        print("🧠 Preparing data for AI prediction...")
        img_size = 48 if self.device.type == 'cpu' else 64
        
        # Generate simple synthetic image based on weather (for AI processing only)
        synthetic_image = np.random.rand(3, img_size, img_size) * 0.3 + 0.4
        
        # Add weather-based patterns to the synthetic image
        cloud_cover = weather_data['cloud_cover'] / 100.0
        humidity = weather_data['humidity'] / 100.0
        
        # Simple cloud simulation for AI input
        for _ in range(max(1, int(cloud_cover * 5))):
            center_x = np.random.randint(5, img_size-5)
            center_y = np.random.randint(5, img_size-5)
            radius = np.random.randint(2, 6)
            
            y, x = np.ogrid[:img_size, :img_size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            cloud_intensity = 0.7 + (cloud_cover * 0.2)
            synthetic_image[:, mask] = np.minimum(cloud_intensity, 0.9)
        
        # Temperature affects brightness
        temp_factor = (weather_data['temperature'] + 10) / 50.0
        temp_factor = np.clip(temp_factor, 0.5, 1.2)
        synthetic_image = synthetic_image * temp_factor
        synthetic_image = np.clip(synthetic_image, 0, 1)        
        # Prepare data for AI prediction
        print("🧠 Making AI prediction...")
        weather_array = np.array([[
            weather_data['temperature'],
            weather_data['humidity'],
            weather_data['pressure'],
            weather_data['wind_speed'],
            weather_data['cloud_cover'],
            weather_data['precipitation'],
            weather_data['visibility'],
            weather_data['uv_index']
        ]])
        
        weather_scaled = self.scaler.transform(weather_array)
        
        # Convert to tensors
        image_tensor = torch.FloatTensor(synthetic_image).unsqueeze(0).to(self.device)
        weather_tensor = torch.FloatTensor(weather_scaled).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(image_tensor, weather_tensor).cpu().item()
        
        # Calculate prediction confidence and trend
        temp_change = prediction - weather_data['temperature']
        if temp_change > 3:
            trend = "🔥 Much warmer"
            trend_emoji = "🔥"
        elif temp_change > 1:
            trend = "🌡️ Warmer"
            trend_emoji = "⬆️"
        elif temp_change < -3:
            trend = "🧊 Much cooler"
            trend_emoji = "🧊"
        elif temp_change < -1:
            trend = "❄️ Cooler"
            trend_emoji = "⬇️"
        else:
            trend = "➡️ Similar"
            trend_emoji = "➡️"
        
        # Weather condition emoji
        if weather_data['cloud_cover'] > 80:
            weather_emoji = "☁️"
        elif weather_data['cloud_cover'] > 50:
            weather_emoji = "⛅"
        elif weather_data['precipitation'] > 0:
            weather_emoji = "🌧️"
        else:
            weather_emoji = "☀️"
        
        # Display comprehensive results
        print("\n" + "🌤️ " * 20)
        print("🎯 WEATHER PREDICTION RESULTS")
        print("🌤️ " * 20)
        
        print(f"\n📍 LOCATION:")
        print(f"   {full_location}")
        print(f"   {lat:.4f}°N, {lon:.4f}°E")
        
        print(f"\n{weather_emoji} CURRENT CONDITIONS:")
        print(f"   🌡️ Temperature: {weather_data['temperature']:.1f}°C")
        print(f"   💧 Humidity: {weather_data['humidity']:.0f}%")
        print(f"   📊 Pressure: {weather_data['pressure']:.0f} hPa")
        print(f"   💨 Wind Speed: {weather_data['wind_speed']:.1f} m/s")
        print(f"   ☁️ Cloud Cover: {weather_data['cloud_cover']:.0f}%")
        if weather_data['precipitation'] > 0:
            print(f"   🌧️ Precipitation: {weather_data['precipitation']:.1f} mm")
        print(f"   👁️ Visibility: {weather_data['visibility']:.1f} km")
        
        if raw_weather:
            weather_desc = raw_weather['weather'][0]['description'].title()
            print(f"   📝 Conditions: {weather_desc}")
        
        print(f"\n🔮 TOMORROW'S AI PREDICTION:")
        print(f"   🌡️ Temperature: {prediction:.1f}°C")
        print(f"   📈 Change: {temp_change:+.1f}°C ({trend})")
        print(f"   🎯 Trend: {trend_emoji}")
        
        # Confidence assessment
        abs_change = abs(temp_change)
        if abs_change < 2:
            confidence = "High"
            conf_emoji = "✅"
        elif abs_change < 4:
            confidence = "Medium"
            conf_emoji = "⚠️"
        else:
            confidence = "Low"
            conf_emoji = "❓"
        print(f"   {conf_emoji} Confidence: {confidence}")
        
        print("\n" + "🌤️ " * 20)
        
        return {
            'location': full_location,
            'coordinates': f"{lat:.4f}°N, {lon:.4f}°E",
            'current_temp': weather_data['temperature'],
            'predicted_temp': prediction,
            'temperature_change': temp_change,
            'trend': trend,
            'confidence': confidence,
            'current_conditions': weather_data,
            'weather_description': raw_weather['weather'][0]['description'] if raw_weather else "Unknown"
        }

def interactive_weather_session():
    """Run an interactive weather prediction session"""
    predictor = InteractiveWeatherPredictor()
    
    if predictor.model is None:
        print("\n❌ Please train the model first by running:")
        print("   python main.py")
        return
    
    print("\n🌍 INTERACTIVE WEATHER PREDICTION SYSTEM")
    print("="*60)
    print("🎯 Enter any place name to get tomorrow's weather prediction!")
    print("💡 Examples: 'New York', 'Mumbai, India', 'Tokyo, Japan'")
    print("🚪 Type 'exit' or 'quit' to stop")
    print("="*60)
    
    while True:
        try:
            # Get user input
            place_name = input("\n🔍 Enter place name: ").strip()
            
            # Check for exit commands
            if place_name.lower() in ['exit', 'quit', 'bye', 'stop', '']:
                print("👋 Thanks for using the Weather Prediction System! Goodbye!")
                break
            
            # Make prediction
            result = predictor.predict_weather_for_place(place_name)
            
            if result:
                # Ask if user wants to continue
                print("\n" + "-"*60)
                continue_choice = input("🔄 Predict weather for another place? (y/n): ").strip().lower()
                if continue_choice in ['n', 'no']:
                    print("👋 Thanks for using the Weather Prediction System! Goodbye!")
                    break
            else:
                print("\n💡 Please try again with a different location name.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the Weather Prediction System! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("💡 Please try again.")

def predict_for_famous_cities():
    """Quick demo with famous cities"""
    predictor = InteractiveWeatherPredictor()
    
    if predictor.model is None:
        print("❌ Model not loaded. Please train first.")
        return
    
    famous_cities = [
        "New York",
        "London", 
        "Tokyo",
        "Mumbai",
        "Sydney",
        "Paris",
        "Dubai"
    ]
    
    print("\n🌍 DEMO: Weather Predictions for Famous Cities")
    print("="*60)
    
    for city in famous_cities:
        print(f"\n{'='*20} {city.upper()} {'='*20}")
        result = predictor.predict_weather_for_place(city)
        if not result:
            print(f"❌ Could not predict weather for {city}")

if __name__ == "__main__":
    # Show menu
    print("\n🌤️ WEATHER PREDICTION SYSTEM")
    print("="*50)
    print("1. Interactive Mode (Enter any place)")
    print("2. Demo Mode (Famous cities)")
    print("3. Exit")
    print("="*50)
    
    choice = input("Choose an option (1-3): ").strip()
    
    if choice == "1":
        interactive_weather_session()
    elif choice == "2":
        predict_for_famous_cities()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Running interactive mode...")
        interactive_weather_session()
