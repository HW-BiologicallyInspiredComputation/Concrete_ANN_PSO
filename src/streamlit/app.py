import streamlit as st
import threading
import time
import random
import pandas as pd
from streamlit_autorefresh import st_autorefresh

from train_manager import TrainManager

st.set_page_config(page_title="AI Trainer Dashboard", layout="wide")

st.title("🤖 AI Training Dashboard")

# Initialize session state and manager
if "manager" not in st.session_state:
    st.session_state.manager = TrainManager()

manager = st.session_state.manager

# Sidebar for training controls
st.sidebar.header("Training Controls")

if not manager.is_training():
    st.sidebar.success("✅ Ready to start training")
    if st.sidebar.button("🚀 Start Training"):
        manager.start_training()
else:
    st.sidebar.warning("⏳ Training is running...")
    if st.sidebar.button("🛑 Stop Training"):
        manager.stop_training()

# Main dashboard
progress = manager.get_progress()
st.subheader("Training Progress")

col1, col2 = st.columns(2)

# Progress bar + metrics
with col1:
    st.progress(progress["percent"])
    st.metric("Current Epoch", progress["epoch"])
    st.metric("Best Fitness", f"{progress['best_fitness']:.4f}")
    st.metric("Elapsed Time (s)", f"{progress['elapsed_time']:.1f}")

# Live graph
with col2:
    st.line_chart(progress["loss_history"])

# Auto refresh every 2 seconds
st_autorefresh(interval=2000, key="training_refresh")
