"""
Simple Weather API - Minimal Version for Fast Deployment
Uses OpenWeatherMap API without AI model for instant deployment
"""

from flask import Flask, render_template, request, jsonify
import requests
import json

app = Flask(__name__)

class SimpleWeatherPredictor:
    def __init__(self):
        self.api_key = "133094071d81a3b5d642d555c8ff0623"
    
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
            print(f"Geocoding error: {e}")
        
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
            print(f"Weather API error: {e}")
        
        return None
    
    def predict_weather_for_place(self, place_name):
        """Simple weather prediction based on current conditions"""
        # Get coordinates
        lat, lon, location_name = self.get_coordinates_from_place(place_name)
        if not lat or not lon:
            return None
        
        # Get current weather
        weather_data = self.get_current_weather(lat, lon)
        if not weather_data:
            return None
        
        current_temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        pressure = weather_data['main']['pressure']
        wind_speed = weather_data['wind']['speed']
        
        # Simple prediction: tomorrow's temp based on current conditions
        # This is a simplified model for fast deployment
        temp_variation = 2.0 if humidity > 70 else 1.0
        predicted_temp = current_temp + temp_variation
        
        # Confidence based on weather stability
        confidence = "High" if abs(temp_variation) < 2 else "Medium"
        
        return {
            'location': location_name,
            'current_temperature': current_temp,
            'predicted_temperature': round(predicted_temp, 1),
            'confidence': confidence,
            'current_conditions': weather_data['weather'][0]['description'],
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed
        }

# Initialize predictor
predictor = SimpleWeatherPredictor()

@app.route('/')
def index():
    """Main page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Weather Prediction System</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
            input { padding: 10px; margin: 10px; width: 200px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; }
            .result { margin-top: 20px; padding: 15px; background: white; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌤️ Weather Prediction System</h1>
            <p>Enter any location to get tomorrow's weather prediction!</p>
            
            <input type="text" id="place" placeholder="Enter city name..." />
            <button onclick="predictWeather()">Get Prediction</button>
            
            <div id="result" class="result" style="display:none;"></div>
        </div>
        
        <script>
            function predictWeather() {
                const place = document.getElementById('place').value;
                if (!place) {
                    alert('Please enter a location');
                    return;
                }
                
                fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({place: place})
                })
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById('result');
                    if (data.success) {
                        const weather = data.data;
                        resultDiv.innerHTML = `
                            <h3>Weather Prediction for ${weather.location}</h3>
                            <p><strong>Current Temperature:</strong> ${weather.current_temperature}°C</p>
                            <p><strong>Tomorrow's Prediction:</strong> ${weather.predicted_temperature}°C</p>
                            <p><strong>Confidence:</strong> ${weather.confidence}</p>
                            <p><strong>Current Conditions:</strong> ${weather.current_conditions}</p>
                            <p><strong>Humidity:</strong> ${weather.humidity}%</p>
                            <p><strong>Wind Speed:</strong> ${weather.wind_speed} m/s</p>
                        `;
                        resultDiv.style.display = 'block';
                    } else {
                        resultDiv.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
                        resultDiv.style.display = 'block';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('result').innerHTML = '<p style="color:red;">An error occurred</p>';
                });
            }
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for weather prediction"""
    try:
        data = request.get_json()
        place_name = data.get('place', '').strip()
        
        if not place_name:
            return jsonify({'error': 'Please enter a place name', 'success': False})
        
        result = predictor.predict_weather_for_place(place_name)
        
        if result:
            return jsonify({'success': True, 'data': result})
        else:
            return jsonify({'error': 'Could not find weather data for this location', 'success': False})
            
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}', 'success': False})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print("🌤️ Simple Weather Prediction Server Starting...")
    print(f"📱 Server running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
