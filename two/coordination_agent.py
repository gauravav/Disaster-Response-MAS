import redis
import math
import time
from datetime import datetime

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
sensor_pred_stream = "sensor_predictions"
tweet_pred_stream = "tweet_predictions"
final_pred_stream = "confirmed_flood_zones"

last_sensor_id = '0'
last_tweet_id = '0'

sensor_preds = []
tweet_preds = []

def within_radius(coord1, coord2, threshold_km=1):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dist = math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111  # approx deg to km
    return dist <= threshold_km

while True:
    sensor_entries = redis_client.xread({sensor_pred_stream: last_sensor_id}, block=5000, count=100)
    tweet_entries = redis_client.xread({tweet_pred_stream: last_tweet_id}, block=5000, count=100)

    for stream, data in sensor_entries:
        last_sensor_id = data[-1][0]
        for entry in data:
            sensor_preds.append((float(entry[1]["lat"]), float(entry[1]["lon"])))

    for stream, data in tweet_entries:
        last_tweet_id = data[-1][0]
        for entry in data:
            tweet_preds.append((float(entry[1]["lat"]), float(entry[1]["lon"])))

    confirmed = []
    for sp in sensor_preds:
        for tp in tweet_preds:
            if within_radius(sp, tp):
                confirmed.append(sp)
                break

    for lat, lon in confirmed:
        redis_client.xadd(final_pred_stream, {
            "source": "coordination_agent",
            "lat": lat,
            "lon": lon,
            "confirmed_by": "sensor+tweet",
            "timestamp": datetime.utcnow().isoformat()
        })

    sensor_preds.clear()
    tweet_preds.clear()
    time.sleep(2)
