import numpy as np
import pandas as pd

def simulate_sensor_data(size=20, flood_spike_zones=None, noise=0.5):
    data = []
    for i in range(size):
        for j in range(size):
            value = np.random.normal(1, noise)
            if flood_spike_zones and (i, j) in flood_spike_zones:
                value += np.random.uniform(3, 6)
            data.append({"x": i, "y": j, "value": round(value, 2)})
    return pd.DataFrame(data)
