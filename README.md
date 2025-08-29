# Fake News Detection AI

A sophisticated web application that uses machine learning to detect fake news and misinformation. Built with Flask, scikit-learn, and modern web technologies.

## Features

- **AI-Powered Detection**: Uses advanced machine learning algorithms to analyze text patterns
- **Real-time Analysis**: Instant results with sophisticated text preprocessing
- **Confidence Scoring**: Detailed probability scores for both fake and real news classifications
- **Beautiful UI**: Modern, responsive design with smooth animations
- **Text Preprocessing**: Advanced NLP techniques including stemming and stopword removal
- **API Endpoints**: RESTful API for integration with other applications

## How It Works

The application uses a **Naive Bayes classifier** trained on a curated dataset of fake and real news examples. Here's the process:

1. **Text Input**: Users input news text, headlines, or articles
2. **Preprocessing**: Text is cleaned, tokenized, and stemmed
3. **Feature Extraction**: TF-IDF vectorization creates numerical features
4. **Classification**: ML model predicts fake vs. real with confidence scores
5. **Results**: Visual display with confidence bars and detailed analysis

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn, NLTK
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Styling**: Custom CSS with gradients and animations
- **Icons**: Font Awesome
- **Deployment**: Gunicorn (production server)

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd fake-news-detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. **Open the Application**: Navigate to the web interface
2. **Input Text**: Paste or type the news text you want to analyze
3. **Click Detect**: The AI will analyze the text in real-time
4. **View Results**: See the classification result with confidence scores
5. **Analyze**: Review the processed text and confidence bars

## API Usage

### Detect Fake News

**Endpoint**: `POST /detect`

**Request Body**:
```json
{
    "text": "Your news text here"
}
```

**Response**:
```json
{
    "is_fake": true,
    "confidence": {
        "fake": 85.5,
        "real": 14.5
    },
    "processed_text": "processed version of input text"
}
```

### Health Check

**Endpoint**: `GET /api/health`

**Response**:
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

## Model Details

- **Algorithm**: Multinomial Naive Bayes
- **Features**: TF-IDF with n-gram range (1,2)
- **Preprocessing**: Lowercase conversion, special character removal, stemming
- **Training Data**: Curated examples of fake and real news
- **Accuracy**: Optimized for demonstration purposes

## Customization

### Adding More Training Data

Edit the `create_model()` function in `app.py` to include more examples:

```python
fake_news_data = [
    "Your fake news examples here",
    # ... more examples
]

real_news_data = [
    "Your real news examples here",
    # ... more examples
]
```

### Changing the Model

You can easily swap the classifier by modifying the model creation:

```python
from sklearn.ensemble import RandomForestClassifier
# or
from sklearn.svm import SVC

# Replace the classifier line
classifier = RandomForestClassifier(n_estimators=100)
```

## Production Deployment

### Using Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Environment Variables

Set these for production:

```bash
export FLASK_ENV=production
export FLASK_DEBUG=0
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application is for educational and demonstration purposes. While it uses machine learning techniques, it should not be considered a definitive source for news verification. Always verify information through multiple reliable sources.

## Future Enhancements

- [ ] Integration with external fact-checking APIs
- [ ] Support for multiple languages
- [ ] Image-based fake news detection
- [ ] User authentication and history
- [ ] Advanced NLP features (sentiment analysis, entity recognition)
- [ ] Real-time news feed monitoring
- [ ] Browser extension for social media platforms
<img width="1150" height="914" alt="Screenshot 2025-08-29 103440" src="https://github.com/user-attachments/assets/a0295569-456f-4417-8cfe-14c5405a8262" />


## Support

For questions or issues, please open an issue on the GitHub repository.

---

**Built with ❤️ using Python and Machine Learning**
