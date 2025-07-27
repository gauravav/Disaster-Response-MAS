import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data.city_grid import create_city_grid
from data.tweets import generate_tweets
from data.sensors import simulate_sensor_data
from simulation.disaster_model_v3 import DisasterModelV3
from simulation.event_queue import event_queue_v3

# ----------------------------------
# Format detected events into DataFrame
def format_event_data(events):
    records = []
    for event in events:
        x, y = event['location']
        records.append({
            'x': x,
            'y': y,
            'type': event['type'],
            'agent': event['agent'],
        })
    return pd.DataFrame(records)
# ----------------------------------

st.set_page_config(layout="wide")
st.title("🚨 Disaster Simulation Dashboard")

# User controls
grid_size = st.slider("City Grid Size", 10, 50, 20)
tweets_per_min = st.slider("Tweets per Minute", 500, 2000, 1000)
misinfo = st.checkbox("Include Misinformation")
sensor_noise = st.slider("Sensor Noise", 0.1, 2.0, 0.5)
sensor_threshold = st.slider("Sensor Threshold", 0.1, 4.0, 0.5)


# Run simulation
if st.button("Run Simulation Step"):
    # Generate input data
    tweets = generate_tweets(num=tweets_per_min, flood_keywords=not misinfo, grid_size=grid_size)
    sensors = simulate_sensor_data(size=grid_size, flood_spike_zones=[(5, 5), (12, 12)], noise=sensor_noise)
    city_grid = create_city_grid(size=grid_size)

    # Run agents
    model = DisasterModelV3(tweets[:200], sensors, sensor_threshold=sensor_threshold)
    model.step()

    # Display events
    st.subheader("🧠 Detected Events")
    st.write(event_queue_v3[:10])

    # Visual Grid using matplotlib
    st.subheader("🗺️ Detected Zones on Grid")

    grid_display = np.zeros((grid_size, grid_size))

    # Mark detected event cells
    for event in event_queue_v3:
        x, y = event['location']
        if 0 <= x < grid_size and 0 <= y < grid_size:
            grid_display[x, y] += 1  # count of events per cell

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.Reds
    cax = ax.imshow(grid_display, cmap=cmap, origin="lower")

    # Add gridlines and labels
    # Label every Nth tick based on grid size
    tick_step = max(grid_size // 10, 1)
    xticks = np.arange(0, grid_size, tick_step)
    yticks = np.arange(0, grid_size, tick_step)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    ax.set_title("Detected Event Locations")
    ax.grid(which='both', color='gray', linewidth=0.5)
    plt.colorbar(cax, ax=ax, label="Event Count")

    st.pyplot(fig)

    # Display all raw data
    with st.expander("📜 All Generated Data"):
        st.subheader("💬 All Tweets")
        st.write(pd.DataFrame(tweets))  # tweets is a list of dicts

        st.subheader("📍 Tweet Coordinates")
        tweet_coords = [tweet['coords'] for tweet in tweets if tweet['coords'] is not None]
        st.write(pd.DataFrame(tweet_coords, columns=['x', 'y']))

        st.subheader("📊 Sensor Readings")
        st.dataframe(sensors)

        st.subheader("🏙️ City Grid")
        st.dataframe(city_grid)
