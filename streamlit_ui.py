import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from matplotlib.colors import ListedColormap

from data.city_grid import create_city_grid

FIGSIZE = (6, 6)

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
    # 3️⃣.1 Land vs Stream Map
    # ---------------------
    st.subheader("🌍 Land (Brown) vs Stream (Light Blue)")

    elevation_map = np.zeros((grid_size, grid_size))
    stream_mask = np.zeros((grid_size, grid_size))

    for _, row in city_grid.iterrows():
        x, y = row["x"], row["y"]
        elevation_map[y, x] = row["elevation"]
        if row["zone_type"] == "stream":
            stream_mask[y, x] = 1

    fig, ax = plt.subplots(figsize=FIGSIZE)
    cmap = cm.get_cmap("YlOrBr")  # continuous brown scale
    elev_img = ax.imshow(elevation_map, cmap=cmap, origin="lower")
    ax.imshow(np.ma.masked_where(stream_mask == 0, stream_mask), cmap=ListedColormap(["#ADD8E6"]), alpha=0.8, origin="lower")

    plt.colorbar(elev_img, ax=ax, label="Elevation")

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Land vs Stream (Initial View)")
    ax.grid(color='gray', linestyle='-', linewidth=0.3)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)

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
    # sim_duration = st.slider("Flood Animation Duration (seconds)", 10, 120, 60)

    if st.button("Start Flood Flow Simulation"):
        import datetime
        start_time = datetime.datetime.now()

        # Add slider for time speed multiplier
        # speed_multiplier = st.slider("⏱️ Time Speed Multiplier", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

        # ✅ Start from stream cells only
        stream_zones = city_grid[city_grid["zone_type"] == "stream"]
        source_zones = stream_zones.sort_values(by="water_level", ascending=False).head(3)[["x", "y"]].values.tolist()

        flooded = set()
        queue = [(x, y) for x, y in source_zones]
        visited = set(queue)
        flood_plot_area = st.empty()
        steps = 0

        max_steps = 3780  # for 1 hour and 3 minutes
        total_runtime_seconds = 3780  # 1 hour and 3 minutes
        sleep_time = total_runtime_seconds / max_steps
        base_spread = 3

        st.info(f"🌀 Simulating flood over 1 hour and 3 minutes from stream zones...")

        while steps < max_steps:
            spread_per_step = base_spread + steps // 300  # increase roughly every 5 minutes
            new_queue = []
            spread_this_step = 0

            for x, y in queue:
                if (x, y) in flooded:
                    continue
                if city_grid[(city_grid["x"] == x) & (city_grid["y"] == y)].iloc[0]["zone_type"] == "stream":
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # cardinal
                else:
                    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # default

                flooded.add((x, y))

                current_cell = city_grid[(city_grid["x"] == x) & (city_grid["y"] == y)].iloc[0]
                current_elev = current_cell["elevation"]
                current_water = current_cell["water_level"]

                neighbors = []
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        if (nx, ny) in visited:
                            continue
                        neighbor_cell = city_grid[(city_grid["x"] == nx) & (city_grid["y"] == ny)].iloc[0]
                        if neighbor_cell["elevation"] < current_elev:
                            neighbors.append(((nx, ny), neighbor_cell["elevation"]))

                neighbors.sort(key=lambda n: n[1])  # prioritize lower elevation first
                for (nx, ny), _ in neighbors:
                    if city_grid[(city_grid["x"] == nx) & (city_grid["y"] == ny)].iloc[0]["zone_type"] != "stream":  # Avoid flooding within stream
                        new_queue.append((nx, ny))
                        visited.add((nx, ny))
                        spread_this_step += 1
                        if spread_this_step >= spread_per_step:
                            break
                if spread_this_step >= spread_per_step:
                    break

            # Visualize current flood
            flood_display_grid = np.zeros((grid_size, grid_size))  # [y, x]
            for _, row in city_grid.iterrows():
                x, y = row["x"], row["y"]
                if row["zone_type"] == "stream":
                    flood_display_grid[y, x] = 1

            for fx, fy in flooded:
                flood_display_grid[fy, fx] = 2  # mark as flooded

            fig, ax = plt.subplots(figsize=FIGSIZE)
            cmap = ListedColormap(["#DEB887", "#ADD8E6", "#00008B"])  # lighter brown, light blue, dark blue
            cax = ax.imshow(flood_display_grid, cmap=cmap, origin="lower")

            tick_step = max(grid_size // 10, 1)
            ax.set_xticks(np.arange(0, grid_size, tick_step))
            ax.set_yticks(np.arange(0, grid_size, tick_step))
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            ax.set_title(f"Elapsed Time: {elapsed_time:.1f}s")
            ax.grid(which='both', color='gray', linewidth=0.5)
            # plt.colorbar(cax, ax=ax, label="Flooded")

            flood_plot_area.pyplot(fig)
            plt.close(fig)

            queue = new_queue or queue
            steps += 1
            time.sleep(sleep_time)

        st.success(f"✅ Flood animation completed in {steps} steps over {total_runtime_seconds} seconds.")

        with st.expander("🗺️ Final Flooded Zone Data"):
            st.dataframe(pd.DataFrame(list(flooded), columns=["x", "y"]))