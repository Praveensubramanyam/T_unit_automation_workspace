# settings.py

class Config:
    DEBUG = True
    TESTING = False
    DATABASE_URI = 'sqlite:///data/database.db'
    SECRET_KEY = 'your_secret_key_here'
    LOGGING_LEVEL = 'INFO'
    MODEL_PATH = 'models/checkpoints/'
    DATA_PATH = {
        'raw': 'data/raw/',
        'processed': 'data/processed/'
    }

class ProductionConfig(Config):
    DEBUG = False
    DATABASE_URI = 'postgresql://user:password@localhost/dbname'
    LOGGING_LEVEL = 'ERROR'

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
    DATABASE_URI = 'sqlite:///data/test_database.db'