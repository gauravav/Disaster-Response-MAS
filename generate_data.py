import numpy as np
import random

def simulate_sensor_data(x, y, flood_zone=False):
    if flood_zone:
        return {"x": x, "y": y, "value": np.random.normal(6, 1)}  # spike
    else:
        return {"x": x, "y": y, "value": np.random.normal(1, 0.5)}

def generate_tweets(num=1000, flood_keywords=True):
    tweets = []
    for _ in range(num):
        if random.random() < 0.3 and flood_keywords:
            tweets.append(f"Help! Water rising fast in zone {random.randint(0, 19)},{random.randint(0, 19)}")
        else:
            tweets.append(random.choice(["Just had lunch", "LOL 😂", "What's up?", "Great weather!"]))
    return tweets
