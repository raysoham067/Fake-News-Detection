"""
Configuration file for the Fake News Detection application
"""

import os

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Application settings
    APP_NAME = 'Fake News Detection AI'
    APP_VERSION = '1.0.0'
    APP_DESCRIPTION = 'Advanced AI-powered tool to detect fake news and misinformation'
    
    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH') or 'fake_news_detector.joblib'
    MAX_TEXT_LENGTH = 10000  # Maximum text length for analysis
    MIN_TEXT_LENGTH = 10     # Minimum text length for analysis
    
    # API settings
    API_RATE_LIMIT = '100 per minute'  # Rate limiting for API endpoints
    
    # Text preprocessing settings
    MAX_FEATURES = 10000     # Maximum features for TF-IDF
    NGRAM_RANGE = (1, 3)     # N-gram range for feature extraction
    MIN_DF = 2               # Minimum document frequency
    MAX_DF = 0.95            # Maximum document frequency
    
    # Classification settings
    CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for classification
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'app.log')
    
    # Security settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    # Performance settings
    WORKERS = int(os.environ.get('WORKERS', 4))
    TIMEOUT = int(os.environ.get('TIMEOUT', 30))

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Override with production values
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable must be set in production")

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """Get configuration class by name"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config.get(config_name, config['default'])
