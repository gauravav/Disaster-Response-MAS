import numpy as np
import pandas as pd

def create_city_grid(size=20, num_stations=6):
    zones = []
    station_types = ["fire", "police", "rescue"]

    # Randomly select unique grid cells to place stations
    total_cells = size * size
    station_indices = np.random.choice(total_cells, num_stations, replace=False)

    for i in range(size):
        for j in range(size):
            index = i * size + j
            is_station = index in station_indices
            zone = {
                "x": i,
                "y": j,
                "zone_type": np.random.choice(["residential", "commercial", "industrial"]),
                "population": np.random.randint(50, 1000),
                "elevation": np.random.uniform(0, 100),
                "water_level": round(np.random.uniform(0, 5), 2),  # in meters
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
                zone["response_time"] = round(np.random.uniform(1.0, 5.0), 2)  # in minutes

            zones.append(zone)

    return pd.DataFrame(zones)
