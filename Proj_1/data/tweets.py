def generate_tweets(num=1000, flood_keywords=True, grid_size=20):
    import random
    tweets = []
    for _ in range(num):
        if random.random() < 0.3 and flood_keywords:
            x = random.randint(0, grid_size - 1)
            y = random.randint(0, grid_size - 1)
            tweet_text = f"Help! Water rising fast in zone {x},{y}"
            tweets.append({'text': tweet_text, 'coords': (x, y)})
        else:
            tweet_text = random.choice([
                "LOL 😂",
                "Nice weather today",
                "Just had lunch",
                "Can’t believe this show!",
                "Running late 😩"
            ])
            tweets.append({'text': tweet_text, 'coords': None})
    return tweets