import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import requests
import json
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI Weather Predictor",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# OpenWeatherMap API Key (Demo mode if invalid)
API_KEY = "133094071d81a3b5d642d555c8ff0623"
DEMO_MODE = False

# Define the neural network models (simplified for Streamlit)
class SimpleWeatherNet(nn.Module):
    def __init__(self):
        super(SimpleWeatherNet, self).__init__()
        # Weather data processing
        self.weather_fc = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, weather_data):
        return self.weather_fc(weather_data)

@st.cache_resource
def load_simple_model():
    """Load or create a simple model for demonstration"""
    model = SimpleWeatherNet()
    
    # Initialize with some reasonable weights for demonstration
    # In a real scenario, you would load a trained model
    with torch.no_grad():
        for param in model.parameters():
            param.data = torch.randn_like(param) * 0.1
    
    model.eval()
    return model

def get_coordinates(city_name):
    """Get latitude and longitude for a city"""
    global DEMO_MODE
    
    if DEMO_MODE:
        return 28.37, 79.45, None  # Default coordinates for demo
    
    geocoding_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={API_KEY}"
    
    try:
        response = requests.get(geocoding_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return None, None, "City not found. Please check the spelling and try again."
        
        return data[0]['lat'], data[0]['lon'], None
    except requests.RequestException as e:
        if "401" in str(e) or "Unauthorized" in str(e):
            DEMO_MODE = True
            return 28.37, 79.45, "API key issue. Using demo mode with sample data."
        return None, None, f"Error getting coordinates: {str(e)}"

def get_weather_data(lat, lon):
    """Get current weather data"""
    global DEMO_MODE
    
    if DEMO_MODE:
        return None, "Using demo mode"
    
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(weather_url, timeout=10)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as e:
        if "401" in str(e) or "Unauthorized" in str(e):
            DEMO_MODE = True
            return None, "API key issue. Using demo mode with sample data."
        return None, f"Error getting weather data: {str(e)}"

def create_demo_weather_data(city_name):
    """Create demo weather data when API is not available"""
    # Sample weather data for demonstration
    demo_data = {
        'name': city_name.title(),
        'sys': {'country': 'XX'},
        'coord': {'lat': 28.37, 'lon': 79.45},
        'main': {
            'temp': np.random.uniform(15, 35),
            'feels_like': np.random.uniform(15, 35),
            'temp_min': np.random.uniform(10, 30),
            'temp_max': np.random.uniform(20, 40),
            'pressure': np.random.uniform(1000, 1020),
            'humidity': np.random.uniform(40, 80)
        },
        'weather': [{'description': 'partly cloudy', 'main': 'Clouds'}],
        'clouds': {'all': np.random.uniform(20, 80)},
        'wind': {'speed': np.random.uniform(2, 15)},
        'visibility': np.random.uniform(8000, 15000)
    }
    return demo_data

def prepare_weather_features(weather_data):
    """Extract and normalize weather features"""
    features = [
        weather_data['main']['temp'],
        weather_data['main']['humidity'],
        weather_data['main']['pressure'],
        weather_data['wind'].get('speed', 0),
        weather_data['clouds']['all'],
        weather_data.get('rain', {}).get('1h', 0),  # precipitation
        weather_data['visibility'] / 1000 if 'visibility' in weather_data else 10,  # visibility in km
        weather_data['coord']['lat']
    ]
    
    # Simple normalization
    normalized_features = [
        (features[0] + 50) / 100,  # temperature
        features[1] / 100,         # humidity
        (features[2] - 900) / 200, # pressure
        features[3] / 20,          # wind speed
        features[4] / 100,         # clouds
        features[5] / 10,          # precipitation
        features[6] / 20,          # visibility
        (features[7] + 90) / 180   # latitude
    ]
    
    return torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0)

def predict_temperature(model, weather_features):
    """Make temperature prediction"""
    with torch.no_grad():
        prediction = model(weather_features)
        return prediction.item()

def create_enhanced_prediction(current_temp, weather_data):
    """Create an enhanced prediction based on weather patterns"""
    # Simple rule-based adjustments for demonstration
    base_change = np.random.normal(0, 2)  # Random daily variation
    
    # Weather-based adjustments
    if weather_data['clouds']['all'] > 80:  # Very cloudy
        base_change -= 1  # Slightly cooler
    elif weather_data['clouds']['all'] < 20:  # Clear skies
        base_change += 0.5  # Slightly warmer
    
    if weather_data.get('rain', {}).get('1h', 0) > 0:  # Rainy
        base_change -= 2  # Cooler when raining
    
    if weather_data['wind'].get('speed', 0) > 10:  # Windy
        base_change -= 0.5  # Wind chill effect
    
    return current_temp + base_change

def main():
    st.title("🌤️ AI Weather Predictor")
    st.markdown("### Get tomorrow's temperature prediction for any city worldwide!")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This AI system uses machine learning to predict tomorrow's temperature "
        "based on current weather conditions. Simply enter a city name and get "
        "an instant prediction!"
    )
    
    st.sidebar.header("How it works")
    st.sidebar.markdown(
        """
        1. 🌍 Enter a city name
        2. 📡 Fetch real-time weather data
        3. 🧠 AI model processes the data
        4. 🌡️ Get tomorrow's temperature prediction
        """
    )
    
    st.sidebar.header("Features")
    st.sidebar.markdown(
        """
        - ✅ Real-time weather data
        - ✅ AI-powered predictions
        - ✅ Global city coverage
        - ✅ Detailed weather info
        - ✅ Fast response time
        """
    )
      # Load model
    try:
        model = load_simple_model()
        st.sidebar.success("✅ AI Model Loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Model Error: {e}")
        model = None
    
    # Demo mode indicator
    if DEMO_MODE:
        st.sidebar.warning("🔄 Demo Mode Active")
    else:
        st.sidebar.success("🌐 Live API Mode")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        city_input = st.text_input(
            "🏙️ Enter city name:",
            placeholder="e.g., London, New York, Tokyo, Mumbai",
            help="Enter any city name worldwide"
        )
        
        predict_button = st.button("🔮 Predict Temperature", type="primary", use_container_width=True)    
    with col2:
        st.markdown("### 📊 Quick Stats")
        st.metric("Model Status", "Active", "✅")
        api_status = "Demo Mode" if DEMO_MODE else "Connected"
        st.metric("API Status", api_status, "🔄" if DEMO_MODE else "🟢")
        st.metric("Response Time", "< 3s", "⚡")
    
    # Prediction logic
    if predict_button and city_input:
        with st.spinner(f"🔍 Getting weather data for {city_input}..."):            # Get coordinates
            lat, lon, error = get_coordinates(city_input)
            
            if error and ("demo mode" in error.lower() or DEMO_MODE):
                # Use demo mode
                st.warning("🔄 Using demo mode with sample data")
                weather_data = create_demo_weather_data(city_input)
                error = None
            elif error:
                st.error(f"❌ {error}")
                return
            else:
                # Get weather data
                weather_data, weather_error = get_weather_data(lat, lon)
                
                if weather_error and ("demo mode" in weather_error.lower() or DEMO_MODE):
                    # Use demo mode
                    st.warning("🔄 Using demo mode with sample data")
                    weather_data = create_demo_weather_data(city_input)
                elif weather_error:
                    st.error(f"❌ {weather_error}")
                    return
        
        with st.spinner("🧠 AI is analyzing the weather patterns..."):
            try:
                # Prepare features
                weather_features = prepare_weather_features(weather_data)
                
                # Make prediction using enhanced method
                current_temp = weather_data['main']['temp']
                
                if model:
                    # Use the neural network for additional processing
                    ai_adjustment = predict_temperature(model, weather_features)
                    predicted_temp = create_enhanced_prediction(current_temp, weather_data) + ai_adjustment * 0.1
                else:
                    # Fallback to rule-based prediction
                    predicted_temp = create_enhanced_prediction(current_temp, weather_data)
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")
                return
        
        # Display results
        st.success("✅ Prediction Complete!")
        
        # Create columns for results
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            temp_change = predicted_temp - current_temp
            change_color = "normal"
            if temp_change > 2:
                change_color = "inverse"
            elif temp_change < -2:
                change_color = "off"
            
            st.metric(
                "🌡️ Tomorrow's Temperature",
                f"{predicted_temp:.1f}°C",
                f"{temp_change:+.1f}°C"
            )
        
        with result_col2:
            st.metric(
                "🌡️ Current Temperature",
                f"{current_temp:.1f}°C"
            )
        
        with result_col3:
            st.metric(
                "🌡️ Feels Like",
                f"{weather_data['main']['feels_like']:.1f}°C"
            )
        
        # Weather summary
        st.subheader("📋 Current Weather Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.write(f"**🏙️ Location:** {weather_data['name']}, {weather_data['sys']['country']}")
            st.write(f"**🌤️ Condition:** {weather_data['weather'][0]['description'].title()}")
            st.write(f"**💧 Humidity:** {weather_data['main']['humidity']}%")
            st.write(f"**🌬️ Wind Speed:** {weather_data['wind'].get('speed', 0)} m/s")
        
        with summary_col2:
            st.write(f"**🌡️ Min Temperature:** {weather_data['main']['temp_min']}°C")
            st.write(f"**🌡️ Max Temperature:** {weather_data['main']['temp_max']}°C")
            st.write(f"**🌊 Pressure:** {weather_data['main']['pressure']} hPa")
            st.write(f"**☁️ Cloudiness:** {weather_data['clouds']['all']}%")
        
        # Additional insights
        st.subheader("🔍 Weather Insights")
        
        insights = []
        
        if weather_data['clouds']['all'] > 80:
            insights.append("☁️ Very cloudy conditions may lead to cooler temperatures")
        elif weather_data['clouds']['all'] < 20:
            insights.append("☀️ Clear skies may result in warmer daytime temperatures")
        
        if weather_data.get('rain', {}).get('1h', 0) > 0:
            insights.append("🌧️ Rain expected, temperatures likely to be cooler")
        
        if weather_data['wind'].get('speed', 0) > 10:
            insights.append("💨 Strong winds may create wind chill effects")
        
        if weather_data['main']['humidity'] > 80:
            insights.append("💧 High humidity may make it feel warmer than actual temperature")
        
        if not insights:
            insights.append("🌟 Weather conditions are moderate and stable")
        
        for insight in insights:
            st.info(insight)
        
        # Timestamp
        st.caption(f"🕐 Prediction generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
          # Disclaimer
        if DEMO_MODE:
            st.caption("⚠️ This is demo mode with sample data. For real predictions, a valid API key is required.")
        else:
            st.caption("⚠️ This is a demonstration model. For critical decisions, consult professional weather services.")
    
    elif predict_button and not city_input:
        st.warning("⚠️ Please enter a city name first!")
    
    # Sample cities
    if not city_input:
        st.subheader("🌍 Try These Popular Cities")
        
        sample_cities = ["London", "New York", "Tokyo", "Mumbai", "Sydney", "Paris", "Cairo", "São Paulo"]
        
        cols = st.columns(4)
        for i, city in enumerate(sample_cities):
            with cols[i % 4]:
                if st.button(city, key=f"sample_{city}"):
                    st.session_state.city_input = city
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "🤖 Powered by AI & Real-time Weather Data | "
        "Made with ❤️ using Streamlit | "
        "Data from OpenWeatherMap API"
        "</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
