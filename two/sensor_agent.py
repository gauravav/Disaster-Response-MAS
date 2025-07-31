import redis
import json
from datetime import datetime
import time

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
sensor_stream = "sensor_data"
prediction_stream = "sensor_predictions"

last_id = '0'

def predict_flood_zones(sensor_data):
    flood_coords = []
    for entry in sensor_data:
        data = entry[1]
        if data.get("is_flooded") == "True" and float(data.get("water_depth", 0)) > 0.5:
            flood_coords.append((float(data["lat"]), float(data["lon"])))
    return flood_coords

print("🔄 Sensor Agent started. Listening for sensor data...")

while True:
    entries = redis_client.xread({sensor_stream: last_id}, block=5000, count=100)
    for stream, data in entries:
        last_id = data[-1][0]
        predictions = predict_flood_zones(data)

        for lat, lon in predictions:
            payload = {
                "source": "sensor_agent",
                "lat": lat,
                "lon": lon,
                "timestamp": datetime.utcnow().isoformat()
            }

            redis_client.xadd(prediction_stream, payload)

            # Print the outgoing data
            print(f"📡 Predicted Flood Location → {payload}")

    time.sleep(1)
