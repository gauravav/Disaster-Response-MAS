import redis
import json
from datetime import datetime
import time

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
tweet_stream = "flood_tweets"
prediction_stream = "tweet_predictions"

last_id = '0'

def predict_flood_from_tweets(tweets):
    predictions = []
    for entry in tweets:
        data = entry[1]
        if data.get("is_genuine") == "True" and float(data.get("flood_severity", 0)) >= 0.6:
            predictions.append((float(data["lat"]), float(data["lon"])))
    return predictions

while True:
    entries = redis_client.xread({tweet_stream: last_id}, block=5000, count=100)
    for stream, data in entries:
        last_id = data[-1][0]
        predictions = predict_flood_from_tweets(data)

        for lat, lon in predictions:
            redis_client.xadd(prediction_stream, {
                "source": "tweets_agent",
                "lat": lat,
                "lon": lon,
                "timestamp": datetime.utcnow().isoformat()
            })

    time.sleep(1)
