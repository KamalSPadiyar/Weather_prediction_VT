# 🌤️ AI Weather Predictor - Streamlit App

A modern, AI-powered weather prediction system that forecasts tomorrow's temperature for any city worldwide using machine learning and real-time weather data.

## 🌟 Features

- **AI-Powered Predictions**: Uses a trained neural network to predict tomorrow's temperature
- **Real-Time Data**: Fetches live weather data from OpenWeatherMap API
- **Global Coverage**: Works for any city worldwide
- **Beautiful UI**: Modern, responsive Streamlit interface
- **Fast & Lightweight**: Optimized for quick deployment and performance

## 🚀 Live Demo

**Deploy on Streamlit Community Cloud:**

1. Fork this repository on GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Set main file path: `streamlit_app.py`
6. Click "Deploy"

Your app will be live at: `https://[your-app-name].streamlit.app`

## 🛠️ Local Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/weather-forecast
cd weather-forecast

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📱 How to Use

1. **Enter City Name**: Type any city name in the input field
2. **Click Predict**: Hit the "Predict Temperature" button
3. **View Results**: Get tomorrow's temperature prediction along with current weather details

## 🧠 How It Works

1. **Geocoding**: Converts city name to coordinates using OpenWeatherMap API
2. **Weather Data**: Fetches current weather conditions
3. **AI Processing**: Neural network analyzes weather patterns
4. **Prediction**: Outputs tomorrow's temperature forecast

## 📋 API Requirements

This app uses the OpenWeatherMap API. The API key is included for demo purposes, but for production use, you should:

1. Get your free API key from [OpenWeatherMap](https://openweathermap.org/api)
2. Replace the API key in `streamlit_app.py`

## 🔧 Technical Details

- **Framework**: Streamlit
- **ML Model**: PyTorch Neural Network
- **APIs**: OpenWeatherMap Geocoding & Weather APIs
- **CPU Optimized**: Designed to run efficiently without GPU

## 📦 Dependencies

- `streamlit>=1.28.0` - Web app framework
- `torch>=2.0.0` - Machine learning model
- `torchvision>=0.15.0` - Computer vision utilities
- `numpy>=1.21.0` - Numerical computing
- `requests>=2.25.0` - HTTP requests

## 🚀 Deployment Options

### Streamlit Community Cloud (Recommended)
- **Free hosting** for public repositories
- **Automatic deployments** from GitHub
- **Custom domain** available
- **Easy sharing** with public URLs

### Alternative Platforms
- **Heroku**: Add `setup.sh` and `Procfile`
- **Render**: Direct deployment from GitHub
- **Railway**: Simple Git-based deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Links

- **Live Demo**: [Deploy on Streamlit](https://share.streamlit.io)
- **API Documentation**: [OpenWeatherMap API](https://openweathermap.org/api)
- **Streamlit Docs**: [Streamlit Documentation](https://docs.streamlit.io)

## 🙋‍♂️ Support

If you have any questions or issues:

1. Check the [Issues](https://github.com/yourusername/weather-forecast/issues) page
2. Create a new issue if needed
3. Contact the maintainers

---

**Made with ❤️ using Streamlit and PyTorch**
