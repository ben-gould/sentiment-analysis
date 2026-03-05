from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "postgresql://mluser:mlpassword@localhost:5432/sentiment_db"
engine = create_engine(DATABASE_URL, echo=False)
session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TrainingData(Base):
    '''Store labeled training data'''

    __tablename__ = "training_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String(20), nullable=False)  # 'positive', 'negative', 'neutral'
    source = Column(String(50))  # 'phrasebank', 'manual', etc.
    created_at = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    '''Store model predictions'''

    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    predicted_sentiment = Column(String(20), nullable=False)
    confidence = Column(Float)
    model_version = Column(String(50))
    latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Model(Base):
    '''Track model versions and metadata'''

    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50))  # 'logistic_regression', 'distilbert', etc.
    f1_score = Column(Float)
    accuracy = Column(Float)
    trained_at = Column(DateTime, default=datetime.utcnow)
    mlflow_run_id = Column(String(100))

class Experiment(Base):
    '''Track MLFlow experiments'''

    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_name = Column(String(100), nullable=False)
    run_id = Column(String(100), unique=True)
    params = Column(Text)  # JSON string of hyperparameters
    metrics = Column(Text)  # JSON string of metrics
    created_at = Column(DateTime, default=datetime.utcnow)

# ============================================================
# ============================================================

def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)
    print("All tables created successfully!")


def get_session():
    """Get a database session."""
    return session_local()


def drop_tables():
    """Drop all tables (use carefully!)."""
    Base.metadata.drop_all(bind=engine)
    print("All tables dropped!")

# ============================================================
# ============================================================

if __name__ == "__main__":
    print('Creating tables...')
    create_tables()

    session = get_session()
    print('Database connection successful.')
    session.close()