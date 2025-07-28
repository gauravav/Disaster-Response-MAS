import numpy as np
import pandas as pd

def create_city_grid(size=20, num_stations=6):
    zones = []
    station_types = ["fire", "police", "rescue"]

    total_cells = size * size
    station_indices = np.random.choice(total_cells, num_stations, replace=False)

    for i in range(size):
        for j in range(size):
            index = i * size + j
            is_station = index in station_indices
            zone = {
                "x": i,
                "y": j,
                "zone_type": "land",  # default
                "population": np.random.randint(50, 1000),
                "elevation": np.random.uniform(0, 100),
                "water_level": round(np.random.uniform(0, 2), 2),
                "station_type": None,
                "dispatch_radius": None,
                "max_teams": None,
                "response_time": None
            }

            if is_station:
                station_type = np.random.choice(station_types)
                zone["station_type"] = station_type
                zone["dispatch_radius"] = np.random.randint(2, 6)
                zone["max_teams"] = np.random.randint(2, 10)
                zone["response_time"] = round(np.random.uniform(1.0, 5.0), 2)

            zones.append(zone)

    # Create a stream path from one edge to another
    stream_cells = set()
    direction = np.random.choice(["top_bottom", "left_right", "diagonal_lr", "diagonal_rl"])

    if direction == "top_bottom":
        x, y = 0, np.random.randint(0, size)
        while x < size:
            stream_cells.add((x, y))
            y += np.random.choice([-1, 0, 1])
            y = max(0, min(size - 1, y))
            x += 1
    elif direction == "left_right":
        x, y = np.random.randint(0, size), 0
        while y < size:
            stream_cells.add((x, y))
            x += np.random.choice([-1, 0, 1])
            x = max(0, min(size - 1, x))
            y += 1
    elif direction == "diagonal_lr":
        x, y = 0, 0
        while x < size and y < size:
            stream_cells.add((x, y))
            x += 1
            y += 1
    elif direction == "diagonal_rl":
        x, y = 0, size - 1
        while x < size and y >= 0:
            stream_cells.add((x, y))
            x += 1
            y -= 1

    for zone in zones:
        if (zone["x"], zone["y"]) in stream_cells:
            zone["zone_type"] = "stream"
            zone["water_level"] = round(np.random.uniform(3.5, 5.0), 2)

    return pd.DataFrame(zones)