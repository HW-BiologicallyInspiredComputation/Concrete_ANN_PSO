# db.py
from sqlalchemy import create_engine, Column, Integer, Float, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import json
from io import StringIO
import os
import time

Base = declarative_base()

class TrainingProgress(Base):
    __tablename__ = "training_progress"
    id = Column(Integer, primary_key=True)
    epoch = Column(Integer, default=0)
    percent = Column(Integer, default=0)
    best_fitness = Column(Float, default=1.0)
    avg_fitness = Column(Float, default=0.0)
    elapsed_time = Column(Float, default=0.0)
    loss_history = Column(String, default="[]")  # store as JSON string
    ga_history = Column(String, default="[]")  # store GA history as JSON
    best_genome = Column(String, default="{}")  # store genome as JSON
    best_genome_timestamp = Column(Float, default=0.0)  # store timestamp when best genome was found

# SQLite engine
engine = create_engine("sqlite:///training.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

def recreate_database():
    """Drop all tables and recreate them"""
    if os.path.exists("training.db"):
        os.remove("training.db")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

# Recreate database with new schema
recreate_database()

# Helpers
def save_progress(progress):
    session = SessionLocal()
    record = session.query(TrainingProgress).first()
    if not record:
        record = TrainingProgress()
    record.epoch = progress["epoch"]
    record.percent = progress["percent"]
    record.best_fitness = progress["best_fitness"]
    record.avg_fitness = progress["avg_fitness"]
    record.elapsed_time = progress["elapsed_time"]
    # convert DataFrames to JSON
    record.loss_history = progress["loss_history"].to_json(orient="records")
    if "ga_history" in progress:
        record.ga_history = progress["ga_history"].to_json(orient="records")
    if "best_genome" in progress and progress["best_genome"]:
        genome = progress["best_genome"]
        print(genome)
        record.best_genome = json.dumps({
            "swarm_size": genome.swarm_size,
            "num_informants": genome.num_informants,
            "ann_layers": genome.ann_layers,
            "particle_initial_position_scale": genome.particle_initial_position_scale,
            "accel": {
                "inertia_weight": genome.accel.inertia_weight,
                "cognitive_weight": genome.accel.cognitive_weight,
                "social_weight": genome.accel.social_weight,
                "global_best_weight": genome.accel.global_best_weight,
                "jump_size": genome.accel.jump_size
            }
        })
        record.best_genome_timestamp = progress.get("best_genome_timestamp", time.time())
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
            "avg_fitness": record.avg_fitness,
            "elapsed_time": record.elapsed_time,
            "loss_history": pd.read_json(StringIO(record.loss_history)) if record.loss_history else pd.DataFrame({"loss":[]}),
            "ga_history": pd.read_json(StringIO(record.ga_history)) if record.ga_history else pd.DataFrame({"best_accuracy":[], "avg_accuracy":[]}),
            "best_genome": json.loads(record.best_genome) if record.best_genome != "{}" else None,
            "best_genome_timestamp": record.best_genome_timestamp
        }
    else:
        progress = {
            "epoch": 0,
            "percent": 0,
            "best_fitness": 1.0,
            "avg_fitness": 0.0,
            "elapsed_time": 0,
            "loss_history": pd.DataFrame({"loss":[]}),
            "ga_history": pd.DataFrame({"best_accuracy":[], "avg_accuracy":[]}),
            "best_genome": None,
            "best_genome_timestamp": 0.0
        }
    session.close()
    return progress
