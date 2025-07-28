import numpy as np
import pandas as pd

def create_city_grid(size=20):
    zones = []
    for i in range(size):
        for j in range(size):
            zones.append({
                "x": i,
                "y": j,
                "zone_type": np.random.choice(["residential", "commercial", "industrial"]),
                "population": np.random.randint(50, 1000),
                "elevation": np.random.uniform(0, 100)
            })
    return pd.DataFrame(zones)
