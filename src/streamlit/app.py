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

# Create two main columns for dashboard layout
dashboard_left, dashboard_right = st.columns(
    [1, 1.5]
)  # Right column slightly wider for chart

with dashboard_left:
    # Sidebar for training controls moved to left column
    st.header("Training Controls")

    if not manager.is_training():
        st.success("✅ Ready to start training")
        if st.button("🚀 Start Training", use_container_width=True):
            manager.start_training()
    else:
        st.warning("⏳ Training is running...")
        if st.button("🛑 Stop Training", use_container_width=True):
            manager.stop_training()

    # Progress and metrics
    progress = manager.get_progress()
    st.subheader("Training Progress")
    st.progress(progress["percent"])
    st.metric("Current Generation", progress["epoch"])
    st.metric("Best Accuracy", f"{progress['best_fitness']:.4f}")
    st.metric("Best Repeats", progress["best_repeats"])
    st.metric("Average Accuracy", f"{progress['avg_fitness']:.4f}")
    st.metric("Elapsed Time (s)", f"{progress['elapsed_time']:.1f}")

with dashboard_right:
    # Chart area
    st.header("Training History")
    chart_placeholder = st.empty()

    if "ga_history" in progress and not progress["ga_history"].empty:
        df = progress["ga_history"].copy()
        chart_placeholder.line_chart(
            data=df,
            y=["best_accuracy", "avg_accuracy"],
            height=400,  # Fixed height for stability
        )

        # Show best hyperparameters
        best_genome = progress.get("best_genome")
        if best_genome:
            timestamp = progress.get("best_genome_timestamp", 0)
            generation = progress["epoch"]
            st.subheader(
                f"🎯 Generation {generation} Best PSO Parameters at {time.strftime('%H:%M:%S', time.localtime(timestamp))}"
            )
            col1, col2 = st.columns(2)
            with col1:
                st.write("**PSO Configuration:**")
                st.write(f"• Swarm Size: {best_genome['swarm_size']}")
                st.write(f"• Number of Informants: {best_genome['num_informants']}")
                st.write(
                    f"• Position Scale: {best_genome['particle_initial_position_scale']}"
                )
                st.write(f"• Topology: {best_genome['ann_layers']}")

            with col2:
                st.write("**Acceleration Coefficients:**")
                accel = best_genome["accel"]
                st.write(f"• Inertia Weight: {accel['inertia_weight']:.3f}")
                st.write(f"• Cognitive Weight: {accel['cognitive_weight']:.3f}")
                st.write(f"• Social Weight: {accel['social_weight']:.3f}")
                st.write(f"• Global Best Weight: {accel['global_best_weight']:.3f}")
                st.write(f"• Jump Size: {accel['jump_size']:.3f}")
    else:
        chart_placeholder.info("Waiting for training data...")

# Reduce refresh rate to prevent flickering
st_autorefresh(interval=3000, key="training_refresh")
