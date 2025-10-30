# db.py
from sqlalchemy import create_engine, Column, Integer, Float, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import json
from io import StringIO

Base = declarative_base()

class TrainingProgress(Base):
    __tablename__ = "training_progress"
    id = Column(Integer, primary_key=True)
    epoch = Column(Integer, default=0)
    percent = Column(Integer, default=0)
    best_fitness = Column(Float, default=1.0)
    elapsed_time = Column(Float, default=0.0)
    loss_history = Column(String, default="[]")  # store as JSON string

# SQLite engine
engine = create_engine("sqlite:///training.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

Base.metadata.create_all(bind=engine)

# Helpers
def save_progress(progress):
    session = SessionLocal()
    record = session.query(TrainingProgress).first()
    if not record:
        record = TrainingProgress()
    record.epoch = progress["epoch"]
    record.percent = progress["percent"]
    record.best_fitness = progress["best_fitness"]
    record.elapsed_time = progress["elapsed_time"]
    # convert DataFrame to JSON
    record.loss_history = progress["loss_history"].to_json(orient="records")
    session.add(record)
    session.commit()
    session.close()

def load_progress():
    session = SessionLocal()
    record = session.query(TrainingProgress).first()
    if record:
        progress = {
            "epoch": record.epoch,
            "percent": record.percent,
            "best_fitness": record.best_fitness,
            "elapsed_time": record.elapsed_time,
            "loss_history": pd.read_json(StringIO(record.loss_history)) if record.loss_history else pd.DataFrame({"loss":[]})
        }
    else:
        progress = {
            "epoch": 0,
            "percent": 0,
            "best_fitness": 1.0,
            "elapsed_time": 0,
            "loss_history": pd.DataFrame({"loss":[]})
        }
    session.close()
    return progress
