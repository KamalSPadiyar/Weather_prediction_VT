"""
Enhanced Weather Prediction System for Any Location
This script allows you to predict weather for any latitude/longitude
"""

import torch
import numpy as np
import pandas as pd
from main import MultiModalWeatherPredictor, get_real_satellite_data, load_trained_model
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime, timedelta

class LocationWeatherPredictor:
    def __init__(self, model_path='best_multimodal_weather_model.pth'):
        """Initialize the weather predictor for any location"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the trained model
        try:
            self.model = MultiModalWeatherPredictor(
                img_size=48 if self.device.type == 'cpu' else 64,
                patch_size=6 if self.device.type == 'cpu' else 8,
                weather_features=8
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("✅ Model loaded successfully!")
        except:
            print("❌ Model not found. Please train the model first by running main.py")
            self.model = None
            return
            
        # Initialize scaler (you'd normally save this from training)
        self.scaler = StandardScaler()
        # Fit with dummy data matching training distribution
        dummy_data = np.array([
            [15, 60, 1013, 5, 50, 2, 10, 5],  # Mean values
            [25, 80, 1033, 10, 80, 5, 15, 8],  # High values
            [5, 40, 993, 0, 20, 0, 5, 2]      # Low values
        ])
        self.scaler.fit(dummy_data)
        
        self.feature_names = ['temperature', 'humidity', 'pressure', 'wind_speed', 
                             'cloud_cover', 'precipitation', 'visibility', 'uv_index']
        
    def get_current_weather_api(self, lat, lon, api_key="133094071d81a3b5d642d555c8ff0623"):
        """
        Get current weather data from OpenWeatherMap API
        """
        import requests
        
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params)
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
    
    def predict_weather(self, lat, lon, location_name=None, api_key="133094071d81a3b5d642d555c8ff0623"):
        """
        Predict tomorrow's temperature for given latitude/longitude
        """
        if self.model is None:
            return None
            
        print(f"\n🌍 Predicting weather for: {location_name or f'{lat:.2f}°N, {lon:.2f}°E'}")
        print("="*60)
        
        # Get current weather data
        print("📡 Fetching current weather data...")
        weather_data, raw_data = self.get_current_weather_api(lat, lon, api_key)
        
        if weather_data is None:
            print("❌ Could not fetch weather data. Using sample data...")
            # Fallback to sample weather data
            weather_data = {
                'temperature': 20,
                'humidity': 65,
                'pressure': 1013,
                'wind_speed': 3,
                'cloud_cover': 40,
                'precipitation': 0,
                'visibility': 10,
                'uv_index': 5
            }
        
        # Get satellite imagery
        print("🛰️ Downloading satellite imagery...")
        try:
            satellite_images, dates = get_real_satellite_data(
                data_source="openweather",
                api_key=api_key,
                lat=lat,
                lon=lon,
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now(),
                img_size=48 if self.device.type == 'cpu' else 64
            )
            
            if len(satellite_images) > 0:
                satellite_image = satellite_images[0]  # Use most recent image
                print("✅ Satellite image downloaded successfully!")
            else:
                raise Exception("No satellite images available")
                
        except Exception as e:
            print(f"⚠️ Could not download satellite image: {e}")
            print("Using synthetic image based on weather conditions...")
            
            # Generate synthetic image based on weather
            img_size = 48 if self.device.type == 'cpu' else 64
            satellite_image = self.generate_weather_based_image(weather_data, img_size)
        
        # Prepare data for model
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
        image_tensor = torch.FloatTensor(satellite_image).unsqueeze(0).to(self.device)
        weather_tensor = torch.FloatTensor(weather_scaled).to(self.device)
        
        # Make prediction
        print("🧠 Making AI prediction...")
        with torch.no_grad():
            prediction = self.model(image_tensor, weather_tensor).cpu().item()
        
        # Display results
        print("\n📊 WEATHER PREDICTION RESULTS")
        print("="*60)
        print(f"📍 Location: {location_name or f'{lat:.2f}°N, {lon:.2f}°E'}")
        if raw_data:
            print(f"🏙️ Area: {raw_data.get('name', 'Unknown')}, {raw_data.get('sys', {}).get('country', '')}")
        
        print(f"\n🌡️ CURRENT CONDITIONS:")
        print(f"   Temperature: {weather_data['temperature']:.1f}°C")
        print(f"   Humidity: {weather_data['humidity']:.0f}%")
        print(f"   Pressure: {weather_data['pressure']:.0f} hPa")
        print(f"   Wind Speed: {weather_data['wind_speed']:.1f} m/s")
        print(f"   Cloud Cover: {weather_data['cloud_cover']:.0f}%")
        print(f"   Precipitation: {weather_data['precipitation']:.1f} mm")
        
        print(f"\n🔮 TOMORROW'S PREDICTION:")
        print(f"   Temperature: {prediction:.1f}°C")
        
        temp_change = prediction - weather_data['temperature']
        if temp_change > 2:
            trend = "🔥 Getting warmer"
        elif temp_change < -2:
            trend = "🧊 Getting cooler"
        else:
            trend = "➡️ Similar temperature"
            
        print(f"   Trend: {trend} ({temp_change:+.1f}°C)")
        
        return {
            'current_temp': weather_data['temperature'],
            'predicted_temp': prediction,
            'temperature_change': temp_change,
            'current_conditions': weather_data,
            'location': location_name or f'{lat:.2f}°N, {lon:.2f}°E'
        }
    
    def generate_weather_based_image(self, weather_data, img_size=48):
        """Generate a synthetic satellite image based on current weather"""
        img = np.random.rand(3, img_size, img_size) * 0.2 + 0.1
        
        # Add clouds based on cloud cover
        cloud_cover = weather_data['cloud_cover'] / 100.0
        n_clouds = int(cloud_cover * 8) + 1
        
        for _ in range(n_clouds):
            center_x = np.random.randint(5, img_size-5)
            center_y = np.random.randint(5, img_size-5)
            radius = np.random.randint(3, 10)
            
            y, x = np.ogrid[:img_size, :img_size]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            cloud_intensity = 0.6 + (cloud_cover * 0.3)
            img[:, mask] = np.minimum(cloud_intensity, 1.0)
        
        return img

def predict_for_multiple_locations():
    """Predict weather for multiple famous locations"""
    predictor = LocationWeatherPredictor()
    
    if predictor.model is None:
        print("Please train the model first by running: python main.py")
        return
    
    # Famous locations
    locations = [
        {"name": "New York City", "lat": 40.7128, "lon": -74.0060},
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
        {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
        {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
        {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
        {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
        {"name": "Delhi", "lat": 28.7041, "lon": 77.1025}
    ]
    
    print("🌍 GLOBAL WEATHER PREDICTIONS")
    print("="*80)
    
    results = []
    for location in locations:
        try:
            result = predictor.predict_weather(
                lat=location["lat"], 
                lon=location["lon"], 
                location_name=location["name"]
            )
            if result:
                results.append(result)
            print()
        except Exception as e:
            print(f"Error predicting for {location['name']}: {e}")
    
    # Summary
    if results:
        print("\n📈 SUMMARY OF PREDICTIONS")
        print("="*80)
        for result in results:
            temp_emoji = "🔥" if result['temperature_change'] > 2 else "🧊" if result['temperature_change'] < -2 else "➡️"
            print(f"{result['location']:20} | Current: {result['current_temp']:5.1f}°C | Tomorrow: {result['predicted_temp']:5.1f}°C | {temp_emoji} {result['temperature_change']:+.1f}°C")

if __name__ == "__main__":
    # Example usage
    predictor = LocationWeatherPredictor()
    
    if predictor.model is None:
        print("❌ Model not found!")
        print("Please run 'python main.py' first to train the model.")
    else:
        print("🎯 SINGLE LOCATION PREDICTION")
        # Predict for a specific location (example: Mumbai)
        result = predictor.predict_weather(lat=19.0760, lon=72.8777, location_name="Mumbai, India")
        
        print("\n" + "="*80)
        print("🌐 MULTIPLE LOCATIONS PREDICTION")
        predict_for_multiple_locations()
