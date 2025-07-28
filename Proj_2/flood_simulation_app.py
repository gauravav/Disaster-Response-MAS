import streamlit as st
import py3dep
import osmnx as ox
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import warnings
import redis
import json
import time
import random
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass
from typing import List, Dict, Tuple
import uuid

warnings.filterwarnings('ignore')

# Set matplotlib backend for Streamlit
plt.switch_backend('Agg')

# Include the FloodTweetGenerator class here (same as above)
@dataclass
class TweetData:
    user_id: str
    username: str
    location: Tuple[float, float]  # (lat, lon)
    text: str
    timestamp: datetime
    is_genuine: bool
    flood_severity: float  # 0-1 scale

class FloodTweetGenerator:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        """Initialize the tweet generator with Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            self.redis_connected = True
        except Exception as e:
            st.warning(f"⚠️ Redis not available: {e}")
            st.info("💡 Install and start Redis for tweet streaming: `redis-server`")
            self.redis_client = None
            self.redis_connected = False

        self.stream_name = "flood_tweets"
        self.is_running = False

        # Tweet templates
        self.normal_tweets = [
            "Beautiful morning in {city}! ☀️",
            "Just grabbed coffee at my favorite spot in {city}",
            "Traffic is moving smoothly today 🚗",
            "Having lunch with friends at downtown {city}",
            "Weather looks nice today in {city}",
            "Great day for a walk in the park!",
            "Just finished work, heading home",
            "Weekend plans anyone? #weekend",
            "Local restaurant has amazing food! 🍕",
            "Enjoying the sunshine today"
        ]

        self.flood_tweets_mild = [
            "Roads are getting a bit wet in {city} 🌧️",
            "Some puddles forming on the streets",
            "Rain is picking up here in {city}",
            "Storm clouds gathering overhead ⛈️",
            "Getting a bit soggy out there",
            "Weather alert just came in for our area",
            "Streets starting to flood a little",
            "Water levels rising in the creek"
        ]

        self.flood_tweets_moderate = [
            "Water rising quickly on {street}! 🚨",
            "Basement starting to flood, need help!",
            "Can't get through downtown {city} - roads flooded",
            "Emergency services are overwhelmed",
            "Water up to my car's tires 😰",
            "Evacuation notice just issued for our area",
            "Power went out, water everywhere!",
            "Need rescue at {street} intersection!"
        ]

        self.flood_tweets_severe = [
            "HELP! Trapped by flood water at {street}! 🆘",
            "Water up to second floor! Need immediate rescue!",
            "EMERGENCY: Family stranded on roof at {street}",
            "Flood waters rising rapidly! Send help NOW!",
            "Can't evacuate - roads completely flooded!",
            "URGENT: Need boat rescue in {city}!",
            "Water everywhere! This is catastrophic! 😱",
            "FLASH FLOOD WARNING! Get out NOW!"
        ]

        self.noise_tweets = [
            "Check out my new profile pic! 📸",
            "Anyone want to trade crypto? 💰",
            "BREAKING: Celebrity scandal rocks Hollywood!",
            "You won't believe this life hack! 🧵Thread",
            "Just posted my workout routine! 💪",
            "New music video just dropped! 🎵",
            "Political drama continues in Washington",
            "Sports score update: Team wins 3-2! ⚽",
            "Fashion week highlights! So trendy! 👗",
            "Tech stocks are up today! 📈",
            "Movie review: Latest blockbuster disappoints",
            "Food trend alert: Everyone's trying this! 🥑"
        ]

        # Generate users
        self.genuine_users = self._generate_users(100, is_genuine=True)
        self.noise_users = self._generate_users(300, is_genuine=False)

    def _generate_users(self, count: int, is_genuine: bool) -> List[Dict]:
        """Generate realistic user profiles"""
        users = []
        first_names = ["Alex", "Jordan", "Casey", "Taylor", "Morgan", "Riley", "Avery",
                       "Quinn", "Blake", "Cameron", "Dana", "Drew", "Finley", "Hayden"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                      "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez"]

        for i in range(count):
            first = random.choice(first_names)
            last = random.choice(last_names)
            username = f"{first.lower()}{last.lower()}{random.randint(1, 999)}"

            users.append({
                'user_id': str(uuid.uuid4()),
                'username': username,
                'full_name': f"{first} {last}",
                'is_genuine': is_genuine
            })
        return users

    def _get_flood_severity(self, lat: float, lon: float, extent: List[float],
                            flood_mask: np.ndarray, elev_array: np.ndarray,
                            water_level: float) -> float:
        """Calculate flood severity at a given location"""
        if not (extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]):
            return 0.0

        lon_idx = int((lon - extent[0]) / (extent[1] - extent[0]) * flood_mask.shape[1])
        lat_idx = int((extent[3] - lat) / (extent[3] - extent[2]) * flood_mask.shape[0])

        lon_idx = max(0, min(flood_mask.shape[1] - 1, lon_idx))
        lat_idx = max(0, min(flood_mask.shape[0] - 1, lat_idx))

        if not flood_mask[lat_idx, lon_idx]:
            return 0.0

        elevation = elev_array[lat_idx, lon_idx]
        water_depth = max(0, water_level - elevation)

        if water_depth <= 0.5:
            return 0.3  # Mild
        elif water_depth <= 2.0:
            return 0.6  # Moderate
        else:
            return 1.0  # Severe

    def generate_tweet(self, extent: List[float], flood_mask: np.ndarray,
                       elev_array: np.ndarray, water_level: float, city_name: str) -> Dict:
        """Generate a single tweet"""
        is_genuine = random.random() < 0.3  # 30% genuine

        # Select user
        user_pool = self.genuine_users if is_genuine else self.noise_users
        user = random.choice(user_pool)

        # Generate location within bounds
        lat = random.uniform(extent[2], extent[3])
        lon = random.uniform(extent[0], extent[1])

        # Calculate flood severity
        severity = self._get_flood_severity(lat, lon, extent, flood_mask, elev_array, water_level)

        # For genuine users during flooding, bias towards flooded areas
        if is_genuine and np.any(flood_mask) and random.random() < 0.7:
            flooded_indices = np.where(flood_mask)
            if len(flooded_indices[0]) > 0:
                idx = random.randint(0, len(flooded_indices[0]) - 1)
                lat_idx, lon_idx = flooded_indices[0][idx], flooded_indices[1][idx]

                # Convert back to lat/lon
                lat = extent[3] - (lat_idx / flood_mask.shape[0]) * (extent[3] - extent[2])
                lon = extent[0] + (lon_idx / flood_mask.shape[1]) * (extent[1] - extent[0])

                # Recalculate severity
                severity = self._get_flood_severity(lat, lon, extent, flood_mask, elev_array, water_level)

        # Select tweet template based on severity
        if not is_genuine:
            template = random.choice(self.noise_tweets)
        elif severity == 0.0:
            template = random.choice(self.normal_tweets)
        elif severity <= 0.4:
            template = random.choice(self.flood_tweets_mild)
        elif severity <= 0.7:
            template = random.choice(self.flood_tweets_moderate)
        else:
            template = random.choice(self.flood_tweets_severe)

        # Fill in location information
        streets = ["Main St", "Oak Ave", "First St", "Park Rd", "River Dr", "Hill St"]
        tweet_text = template.format(city=city_name, street=random.choice(streets))

        return {
            'user_id': user['user_id'],
            'username': user['username'],
            'lat': lat,
            'lon': lon,
            'text': tweet_text,
            'timestamp': datetime.now().isoformat(),
            'is_genuine': is_genuine,
            'flood_severity': severity
        }

    def add_tweet_to_stream(self, tweet: Dict):
        """Add tweet to Redis stream or session state"""
        if self.redis_connected:
            try:
                tweet_data = {k: str(v) for k, v in tweet.items()}
                message_id = self.redis_client.xadd(self.stream_name, tweet_data)
                return message_id
            except Exception as e:
                st.error(f"Redis error: {e}")
                return False
        else:
            # Fallback to session state
            if 'tweets' not in st.session_state:
                st.session_state.tweets = []
            st.session_state.tweets.append(tweet)
            # Keep only last 100 tweets
            st.session_state.tweets = st.session_state.tweets[-100:]
            return True

    def get_recent_tweets(self, count: int = 20) -> List[Dict]:
        """Get recent tweets from Redis or session state"""
        if self.redis_connected:
            try:
                messages = self.redis_client.xrevrange(self.stream_name, count=count)
                tweets = []
                for message_id, fields in messages:
                    tweet = {
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
            except Exception:
                return []
        else:
            # Fallback to session state
            tweets = st.session_state.get('tweets', [])
            return tweets[-count:] if tweets else []

    def clear_stream(self):
        """Clear all tweets"""
        if self.redis_connected:
            try:
                self.redis_client.delete(self.stream_name)
                return True
            except Exception:
                return False
        else:
            st.session_state.tweets = []
            return True

# -------------------------------
# Streamlit Setup
st.set_page_config(layout="wide")
st.title("🌊 Flood Simulation with Real-Time Social Media Feed")

# Initialize tweet generator
if 'tweet_generator' not in st.session_state:
    st.session_state.tweet_generator = FloodTweetGenerator()

# -------------------------------
# User Inputs
st.sidebar.header("🌊 Flood Simulation Settings")
place = st.sidebar.text_input("Enter US Location", "Dallas, Texas, USA")
water_level = st.sidebar.slider("Flood Water Level (m above sea level)", 0, 300, 100)
show_rivers = st.sidebar.checkbox("Overlay Rivers", True)
show_buildings = st.sidebar.checkbox("Overlay Buildings", False)

st.sidebar.header("🐦 Tweet Stream Settings")
tweet_enabled = st.sidebar.checkbox("Enable Tweet Stream", True)
tweet_rate = st.sidebar.slider("Tweets per minute", 1, 30, 10)

if st.sidebar.button("🗑️ Clear All Tweets"):
    st.session_state.tweet_generator.clear_stream()
    st.sidebar.success("Tweets cleared!")

# -------------------------------
# Get bounding box
try:
    place_gdf = ox.geocode_to_gdf(place)
    place_geom = place_gdf.geometry[0]
    bounds = place_geom.bounds

    osm_bbox = (bounds[3], bounds[1], bounds[2], bounds[0])
    py3dep_bbox = (bounds[1], bounds[0], bounds[3], bounds[2])

    st.success(f"📍 Location: {place}")

    # Store in session state
    st.session_state.place = place
    st.session_state.bounds = bounds

except Exception as e:
    st.error(f"Location not found: {e}")
    st.stop()

# -------------------------------
# Fetch elevation data (same as before)
with st.spinner("Downloading elevation data..."):
    try:
        st.info("Downloading elevation data...")

        try:
            dem_data = py3dep.get_dem(geometry=place_geom, resolution=30, crs="EPSG:4326")
            elev_array = dem_data.values.squeeze()
            dem_bounds = dem_data.rio.bounds()
            extent = [dem_bounds[0], dem_bounds[2], dem_bounds[1], dem_bounds[3]]
            st.success("✅ Elevation data downloaded")
        except Exception as e1:
            try:
                bbox_simple = [bounds[1], bounds[0], bounds[3], bounds[2]]
                dem_data = py3dep.get_dem(bbox_simple, resolution=90)
                elev_array = dem_data.squeeze()
                extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
                st.success("✅ Elevation data downloaded (fallback method)")
            except Exception as e2:
                st.error(f"All elevation methods failed: {e1}, {e2}")
                st.stop()

        # Validate and clean data
        if elev_array is None or elev_array.size == 0:
            st.error("No elevation data available")
            st.stop()

        nan_count = np.isnan(elev_array).sum()
        if nan_count == elev_array.size:
            st.error("All elevation values are invalid")
            st.stop()
        elif nan_count > 0:
            elev_array = np.where(np.isnan(elev_array), np.nanmean(elev_array), elev_array)

        # Store in session state for tweet generation
        st.session_state.elev_array = elev_array
        st.session_state.extent = extent
        st.session_state.water_level = water_level

        st.info(f"Elevation range: {np.nanmin(elev_array):.1f}m to {np.nanmax(elev_array):.1f}m")

    except Exception as e:
        st.error(f"Critical error: {e}")
        st.stop()

# -------------------------------
# Flood Simulation
flood_mask = elev_array <= water_level
flooded_area_pct = (np.sum(flood_mask) / flood_mask.size) * 100
st.session_state.flood_mask = flood_mask

st.info(f"💧 Flooded area: {flooded_area_pct:.1f}% of the region")

# -------------------------------
# Create layout with map and tweets side by side
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ Flood Simulation Map")

    # Generate tweets if enabled
    if tweet_enabled:
        city_name = place.split(',')[0]
        for _ in range(random.randint(1, 3)):
            tweet = st.session_state.tweet_generator.generate_tweet(
                extent, flood_mask, elev_array, water_level, city_name
            )
            st.session_state.tweet_generator.add_tweet_to_stream(tweet)

    # Plotting (simplified version)
    with st.spinner("Creating map..."):
        try:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

            # Plot elevation
            elev_img = ax.imshow(elev_array, cmap='terrain', extent=extent, origin='upper', alpha=0.9)
            fig.colorbar(elev_img, ax=ax, label="Elevation (m)", shrink=0.8)

            # Plot flood overlay
            if np.any(flood_mask):
                flood_overlay = np.ma.masked_where(~flood_mask, np.ones_like(flood_mask))
                ax.imshow(flood_overlay, cmap='Blues', alpha=0.6, extent=extent, origin='upper')

            # Add coordinate system
            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
            ax.set_title(f"Flood Simulation: {place}\nWater Level: {water_level}m | Flooded: {flooded_area_pct:.1f}%")
            ax.grid(True, alpha=0.3)

            st.pyplot(fig, clear_figure=True, use_container_width=True)
            plt.close(fig)

        except Exception as e:
            st.error(f"Map rendering error: {e}")

with col2:
    st.subheader("🐦 Live Tweet Feed")

    if tweet_enabled:
        # Tweet stream status
        if st.session_state.tweet_generator.redis_connected:
            st.success("🟢 Redis Connected")
        else:
            st.info("🟡 Using Local Storage")

        # Get and display recent tweets
        recent_tweets = st.session_state.tweet_generator.get_recent_tweets(15)

        if recent_tweets:
            # Calculate statistics
            genuine_count = sum(1 for t in recent_tweets if t['is_genuine'])
            noise_count = len(recent_tweets) - genuine_count

            st.metric("Recent Tweets", len(recent_tweets))
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Genuine", genuine_count)
            with col_b:
                st.metric("Noise", noise_count)

            st.write("---")

            # Display tweets with color coding
            for tweet in reversed(recent_tweets[-10:]):  # Show latest 10
                timestamp = tweet['timestamp'][:19] if isinstance(tweet['timestamp'], str) else str(tweet['timestamp'])[:19]

                if tweet['is_genuine']:
                    if tweet['flood_severity'] > 0.7:
                        st.error(f"🆘 **@{tweet['username']}**\n{tweet['text']}\n*{timestamp}*")
                    elif tweet['flood_severity'] > 0.4:
                        st.warning(f"⚠️ **@{tweet['username']}**\n{tweet['text']}\n*{timestamp}*")
                    elif tweet['flood_severity'] > 0:
                        st.info(f"🌧️ **@{tweet['username']}**\n{tweet['text']}\n*{timestamp}*")
                    else:
                        st.success(f"☀️ **@{tweet['username']}**\n{tweet['text']}\n*{timestamp}*")
                else:
                    with st.container():
                        st.write(f"📱 **@{tweet['username']}** (noise)\n{tweet['text']}\n*{timestamp}*")

                st.write("")  # Add spacing
        else:
            st.info("No tweets yet. Refresh to see new tweets!")
    else:
        st.info("Tweet stream disabled. Enable in sidebar to see social media activity.")

# -------------------------------
# Bottom statistics
st.subheader("📊 Simulation Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Water Level", f"{water_level} m")
with col2:
    st.metric("Flooded Area", f"{flooded_area_pct:.1f}%")
with col3:
    st.metric("Elevation Range", f"{np.nanmin(elev_array):.0f}m - {np.nanmax(elev_array):.0f}m")
with col4:
    tweets_count = len(st.session_state.tweet_generator.get_recent_tweets(100))
    st.metric("Total Tweets", tweets_count)

st.warning("⚠️ This is a simplified flood simulation for educational purposes only. Real flood modeling requires additional factors like rainfall, drainage, soil permeability, and temporal dynamics.")

# Simple auto-refresh mechanism for tweets
if tweet_enabled:
    # Initialize auto-refresh counter
    if 'auto_refresh_counter' not in st.session_state:
        st.session_state.auto_refresh_counter = 0

    # Auto-generate tweets periodically
    st.session_state.auto_refresh_counter += 1
    if st.session_state.auto_refresh_counter % 5 == 0:  # Every 5th page interaction
        city_name = place.split(',')[0]
        tweet = st.session_state.tweet_generator.generate_tweet(
            extent, flood_mask, elev_array, water_level, city_name
        )
        st.session_state.tweet_generator.add_tweet_to_stream(tweet)

    # Add an auto-refresh toggle in sidebar
    auto_refresh = st.sidebar.checkbox("Auto-refresh tweets", value=True)
    if auto_refresh:
        time.sleep(3)
        st.rerun()