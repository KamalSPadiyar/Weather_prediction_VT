"""
Web Version of Weather Prediction System
Simple Flask web interface for the weather predictor
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from interactive_weather import InteractiveWeatherPredictor

app = Flask(__name__)

# Initialize the weather predictor
try:
    predictor = InteractiveWeatherPredictor()
    model_loaded = predictor.model is not None
except Exception as e:
    predictor = None
    model_loaded = False
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for weather prediction"""
    try:
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded. Please ensure the model file exists.',
                'success': False
            })
        
        data = request.get_json()
        place_name = data.get('place', '').strip()
        
        if not place_name:
            return jsonify({
                'error': 'Please enter a place name',
                'success': False
            })
        
        # Get prediction
        result = predictor.predict_weather_for_place(place_name)
        
        if result:
            return jsonify({
                'success': True,
                'data': result
            })
        else:
            return jsonify({
                'error': 'Could not find weather data for this location',
                'success': False
            })
            
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'success': False
        })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'version': '1.0'
    })

if __name__ == '__main__':
    print("🌤️ Weather Prediction Web Server")
    print("=" * 40)
    
    # Get port from environment variable (for cloud deployment)
    import os
    port = int(os.environ.get('PORT', 5000))
    
    if model_loaded:
        print("✅ AI Model loaded successfully")
        print("🌐 Starting web server...")
        print(f"📱 Server will be available on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("❌ Model not loaded - please check model file")
        print("Make sure 'best_multimodal_weather_model.pth' exists")
        # Still start the server to show error page
        app.run(host='0.0.0.0', port=port, debug=False)
