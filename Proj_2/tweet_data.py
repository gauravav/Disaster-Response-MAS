import redis
import json
import pandas as pd
from datetime import datetime

def read_flood_tweets(count=50):
    """Read tweet data from Redis stream"""
    try:
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

        # Read latest tweets
        messages = r.xrevrange('flood_tweets', count=count)

        tweets = []
        for message_id, fields in messages:
            tweet = {
                'message_id': message_id,
                'user_id': fields.get('user_id', ''),
                'username': fields.get('username', ''),
                'lat': float(fields.get('lat', 0)),
                'lon': float(fields.get('lon', 0)),
                'text': fields.get('text', ''),
                'timestamp': fields.get('timestamp', ''),
                'is_genuine': fields.get('is_genuine', 'False') == 'True',
                'flood_severity': float(fields.get('flood_severity', 0))
            }
            tweets.append(tweet)

        return tweets

    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        return []

def analyze_tweets(tweets):
    """Analyze tweet data"""
    if not tweets:
        print("No tweets found")
        return

    df = pd.DataFrame(tweets)

    print(f"\n📊 TWEET ANALYSIS")
    print(f"Total tweets: {len(tweets)}")
    print(f"Genuine tweets: {df['is_genuine'].sum()}")
    print(f"Noise tweets: {(~df['is_genuine']).sum()}")
    print(f"Genuine percentage: {(df['is_genuine'].sum() / len(tweets) * 100):.1f}%")

    # Severity distribution
    severity_counts = {
        'Normal (0)': (df['flood_severity'] == 0).sum(),
        'Mild (0.1-0.4)': ((df['flood_severity'] > 0) & (df['flood_severity'] <= 0.4)).sum(),
        'Moderate (0.4-0.7)': ((df['flood_severity'] > 0.4) & (df['flood_severity'] <= 0.7)).sum(),
        'Severe (0.7+)': (df['flood_severity'] > 0.7).sum()
    }

    print(f"\n🌊 FLOOD SEVERITY DISTRIBUTION:")
    for severity, count in severity_counts.items():
        print(f"  {severity}: {count}")

    return df

def display_recent_tweets(tweets, limit=10):
    """Display recent tweets"""
    print(f"\n🐦 RECENT TWEETS (last {limit}):")
    print("-" * 80)

    for i, tweet in enumerate(tweets[:limit]):
        genuine_marker = "✅ GENUINE" if tweet['is_genuine'] else "❌ NOISE"
        severity = tweet['flood_severity']
        severity_text = (
            "🟢 Normal" if severity == 0 else
            "🟡 Mild" if severity <= 0.4 else
            "🟠 Moderate" if severity <= 0.7 else
            "🔴 SEVERE"
        )

        print(f"{i+1:2d}. @{tweet['username']} {genuine_marker} {severity_text}")
        print(f"    📍 ({tweet['lat']:.4f}, {tweet['lon']:.4f})")
        print(f"    💬 {tweet['text']}")
        print(f"    🕒 {tweet['timestamp']}")
        print()

def monitor_live_tweets():
    """Monitor tweets in real-time"""
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

        print("🔴 MONITORING LIVE TWEETS (Ctrl+C to stop)")
        print("-" * 50)

        # Start from latest message
        last_id = '$'

        while True:
            # Block and wait for new messages
            streams = r.xread({'flood_tweets': last_id}, block=1000)

            for stream, messages in streams:
                for message_id, fields in messages:
                    genuine_marker = "✅" if fields.get('is_genuine') == 'True' else "❌"
                    severity = float(fields.get('flood_severity', 0))
                    severity_text = (
                        "🟢" if severity == 0 else
                        "🟡" if severity <= 0.4 else
                        "🟠" if severity <= 0.7 else
                        "🔴"
                    )

                    print(f"{genuine_marker} {severity_text} @{fields.get('username')}: {fields.get('text')}")
                    last_id = message_id

    except KeyboardInterrupt:
        print("\n⏹️  Stopped monitoring")
    except Exception as e:
        print(f"Error: {e}")

def export_tweets_to_csv(tweets, filename="flood_tweets.csv"):
    """Export tweets to CSV file"""
    if tweets:
        df = pd.DataFrame(tweets)
        df.to_csv(filename, index=False)
        print(f"📁 Exported {len(tweets)} tweets to {filename}")
    else:
        print("No tweets to export")

if __name__ == "__main__":
    print("🌊 FLOOD TWEET ANALYZER")
    print("=" * 30)

    # Read tweets
    tweets = read_flood_tweets(count=100)

    if tweets:
        # Analyze tweets
        df = analyze_tweets(tweets)

        # Display recent tweets
        display_recent_tweets(tweets, limit=5)

        # Export option
        export_choice = input("\nExport tweets to CSV? (y/n): ")
        if export_choice.lower() == 'y':
            export_tweets_to_csv(tweets)

        # Live monitoring option
        live_choice = input("\nMonitor live tweets? (y/n): ")
        if live_choice.lower() == 'y':
            monitor_live_tweets()

    else:
        print("❌ No tweets found. Make sure:")
        print("  1. Redis is running")
        print("  2. The Streamlit app is generating tweets")
        print("  3. Tweet streaming is enabled")