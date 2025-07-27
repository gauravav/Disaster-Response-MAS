import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

from data.city_grid import create_city_grid

st.set_page_config(layout="wide")
st.title("🏙️ Elevation-Based City Flood Simulation")

# ---------------------
# 1️⃣ Settings
# ---------------------
grid_size = st.slider("City Grid Size", 10, 50, 20)
num_stations = st.slider("Number of Emergency Stations", 1, 10, 5)

# Persistent storage for grid
if "city_grid" not in st.session_state:
    st.session_state["city_grid"] = None

# ---------------------
# 2️⃣ Generate City Button
# ---------------------
if st.button("🚀 Generate City"):
    city_grid = create_city_grid(size=grid_size, num_stations=num_stations)
    st.session_state["city_grid"] = city_grid

# Retrieve current city grid
city_grid = st.session_state["city_grid"]

if city_grid is not None:
    # ---------------------
    # 3️⃣ Elevation Map
    # ---------------------
    st.subheader("🌄 Elevation Map with Emergency Stations")

    fig, ax = plt.subplots(figsize=(6, 6))
    elevation_grid = np.zeros((grid_size, grid_size))

    for _, row in city_grid.iterrows():
        elevation_grid[row["x"], row["y"]] = row["elevation"]

    cmap = plt.cm.YlOrBr
    elev_img = ax.imshow(elevation_grid, cmap=cmap, origin="upper")

    for _, row in city_grid.iterrows():
        x, y = row["x"], row["y"]
        station = row["station_type"]
        if station:
            icon = {"fire": "🔥", "police": "👮", "rescue": "🚑"}[station]
            ax.text(y, x + 0.5, icon, ha="center", va="center", fontsize=7)

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("City Elevation Map")
    ax.grid(color='gray', linestyle='-', linewidth=0.3)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)

    cbar = plt.colorbar(elev_img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Elevation")

    st.pyplot(fig)

    # ---------------------
    # 4️⃣ City Grid Table
    # ---------------------
    with st.expander("📋 View City Data Table"):
        st.dataframe(city_grid)

    # ---------------------
    # 5️⃣ Simulate Flood Flow (Live)
    # ---------------------
    st.subheader("🌊 Simulate Flood Flow (Based on Elevation + Water Level)")
    sim_duration = st.slider("Flood Animation Duration (seconds)", 10, 120, 60)

    if st.button("Start Flood Flow Simulation"):
        sorted_zones = city_grid.sort_values(by=["water_level", "elevation"], ascending=[False, False])
        source_zones = sorted_zones.head(3)[["x", "y"]].values.tolist()

        flooded = set()
        queue = [(x, y) for x, y in source_zones]
        visited = set(queue)
        flood_plot_area = st.empty()
        steps = 0

        max_steps = 100
        sleep_time = sim_duration / max_steps
        spread_per_step = max(1, (grid_size * grid_size) // max_steps)

        st.info(f"🌀 Simulating flood over {sim_duration} seconds based on water levels...")

        while steps < max_steps:
            new_queue = []
            spread_this_step = 0

            for x, y in queue:
                if (x, y) in flooded:
                    continue
                flooded.add((x, y))

                current_cell = city_grid[(city_grid["x"] == x) & (city_grid["y"] == y)].iloc[0]
                current_elev = current_cell["elevation"]
                current_water = current_cell["water_level"]

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        if (nx, ny) in visited:
                            continue
                        neighbor_cell = city_grid[(city_grid["x"] == nx) & (city_grid["y"] == ny)].iloc[0]
                        if (neighbor_cell["elevation"] < current_elev or
                                neighbor_cell["water_level"] < current_water):
                            new_queue.append((nx, ny))
                            visited.add((nx, ny))
                            spread_this_step += 1
                            if spread_this_step >= spread_per_step:
                                break
                if spread_this_step >= spread_per_step:
                    break

            # Visualize current flood
            flood_grid = np.zeros((grid_size, grid_size))
            for fx, fy in flooded:
                flood_grid[fx, fy] = 1

            fig, ax = plt.subplots(figsize=(6, 6))
            cax = ax.imshow(flood_grid, cmap=plt.cm.Blues, origin="lower")

            tick_step = max(grid_size // 10, 1)
            ax.set_xticks(np.arange(0, grid_size, tick_step))
            ax.set_yticks(np.arange(0, grid_size, tick_step))
            ax.set_title(f"Flood Step {steps + 1} / {max_steps}")
            ax.set_xlabel("Y")
            ax.set_ylabel("X")
            ax.grid(which='both', color='gray', linewidth=0.5)
            plt.colorbar(cax, ax=ax, label="Flooded")

            flood_plot_area.pyplot(fig)

            queue = new_queue or queue
            steps += 1
            time.sleep(sleep_time)

        st.success(f"✅ Flood animation completed in {steps} steps over {sim_duration} seconds.")

        with st.expander("🗺️ Final Flooded Zone Data"):
            st.dataframe(pd.DataFrame(list(flooded), columns=["x", "y"]))
