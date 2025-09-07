"""
Simple Weather Predictor - Just enter a place name!
"""

from interactive_weather import InteractiveWeatherPredictor

def simple_weather_prediction():
    """Simple function to predict weather for any place"""
    print("\n🌤️ WEATHER PREDICTOR")
    print("="*40)
    print("Enter any place name to get tomorrow's weather prediction!")
    print("Examples: 'Mumbai', 'New York', 'London', 'Tokyo'")
    print("="*40)
    
    # Initialize predictor
    predictor = InteractiveWeatherPredictor()
    
    if predictor.model is None:
        print("❌ Please train the model first by running: python main.py")
        return
    
    while True:
        # Get place name from user
        place_name = input("\n🌍 Enter place name (or 'exit' to quit): ").strip()
        
        if place_name.lower() in ['exit', 'quit', 'bye', '']:
            print("👋 Goodbye!")
            break
        
        # Make prediction
        print(f"\n🔍 Predicting weather for: {place_name}")
        result = predictor.predict_weather_for_place(place_name)
        
        if not result:
            print("❌ Location not found. Please try again.")

if __name__ == "__main__":
    simple_weather_prediction()
