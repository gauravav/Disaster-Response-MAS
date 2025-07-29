import streamlit as st
import py3dep
import osmnx as ox
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
import pandas as pd

warnings.filterwarnings('ignore')

# Set matplotlib backend for Streamlit
plt.switch_backend('Agg')

@dataclass
class TweetData:
    user_id: str
    username: str
    location: Tuple[float, float]  # (lat, lon)
    text: str
    timestamp: datetime
    is_genuine: bool
    flood_severity: float  # 0-1 scale

class LocalFloodManager:
    def __init__(self):
        """Initialize local flood event manager"""
        self.flood_events = []

    def add_flood_event(self, lat: float, lon: float, water_level: float, radius_km: float = 2.0):
        """Add a localized flood event"""
        flood_event = {
            'id': f'FLOOD_{len(self.flood_events)+1:03d}',
            'lat': lat,
            'lon': lon,
            'water_level': water_level,
            'radius_km': radius_km,
            'created_at': datetime.now(),
            'active': True
        }
        self.flood_events.append(flood_event)
        return flood_event

    def remove_flood_event(self, event_id: str):
        """Remove a flood event"""
        self.flood_events = [event for event in self.flood_events if event['id'] != event_id]

    def clear_all_floods(self):
        """Clear all local flood events"""
        self.flood_events = []

    def calculate_combined_flood_mask(self, extent: List[float], elev_array: np.ndarray,
                                      global_water_level: float) -> np.ndarray:
        """Calculate flood mask combining global and local floods"""
        # Start with global flood mask
        global_flood_mask = elev_array <= global_water_level

        # If no local floods, return global mask
        if not self.flood_events:
            return global_flood_mask

        # Create combined mask
        combined_mask = global_flood_mask.copy()

        # Add local flood effects
        for event in self.flood_events:
            if not event['active']:
                continue

            # Convert radius from km to degrees (approximate)
            radius_deg = event['radius_km'] / 111.0  # 1 degree ≈ 111 km

            # Create meshgrid for distance calculation
            lat_array = np.linspace(extent[3], extent[2], elev_array.shape[0])
            lon_array = np.linspace(extent[0], extent[1], elev_array.shape[1])
            lon_grid, lat_grid = np.meshgrid(lon_array, lat_array)

            # Calculate distance from flood center
            distance = np.sqrt((lat_grid - event['lat'])**2 + (lon_grid - event['lon'])**2)

            # Apply local flooding within radius
            local_flood_area = distance <= radius_deg
            local_flood_mask = (elev_array <= event['water_level']) & local_flood_area

            # Combine with existing mask
            combined_mask = combined_mask | local_flood_mask

        return combined_mask

class FloodSensorNetwork:
    def __init__(self, num_sensors: int = 20):
        """Initialize flood sensor network with water level sensors only"""
        self.sensors = []
        self.num_sensors = num_sensors
        self.last_stream_time = datetime.now()

    def deploy_sensors(self, extent: List[float], elev_array: np.ndarray) -> List[Dict]:
        """Deploy water level sensors randomly across the map area"""
        sensors = []

        # Only water level sensors
        sensor_info = {'icon': '🌊', 'color': 'blue', 'baseline': 0.1}

        for i in range(self.num_sensors):
            # Random location within bounds
            lat = random.uniform(extent[2], extent[3])
            lon = random.uniform(extent[0], extent[1])

            # Get elevation at sensor location
            lon_idx = int((lon - extent[0]) / (extent[1] - extent[0]) * elev_array.shape[1])
            lat_idx = int((extent[3] - lat) / (extent[3] - extent[2]) * elev_array.shape[0])

            lon_idx = max(0, min(elev_array.shape[1] - 1, lon_idx))
            lat_idx = max(0, min(elev_array.shape[0] - 1, lat_idx))

            elevation = elev_array[lat_idx, lon_idx] if lat_idx < elev_array.shape[0] else np.nanmean(elev_array)

            sensor = {
                'id': f'SENSOR_{i+1:03d}',
                'type': 'water_level',
                'lat': lat,
                'lon': lon,
                'elevation': elevation,
                'icon': sensor_info['icon'],
                'color': sensor_info['color'],
                'baseline_reading': sensor_info['baseline'],
                'current_reading': sensor_info['baseline'],
                'status': 'operational',
                'last_update': datetime.now(),
                'alert_level': 'normal',
                'is_flooded': False
            }
            sensors.append(sensor)

        self.sensors = sensors
        return sensors

    def update_sensor_readings(self, extent: List[float], flood_mask: np.ndarray,
                               elev_array: np.ndarray, water_level: float) -> List[Dict]:
        """Update water level sensor readings based on current flood conditions"""
        updated_sensors = []

        for sensor in self.sensors:
            # Calculate if sensor location is flooded
            lon_idx = int((sensor['lon'] - extent[0]) / (extent[1] - extent[0]) * flood_mask.shape[1])
            lat_idx = int((extent[3] - sensor['lat']) / (extent[3] - extent[2]) * flood_mask.shape[0])

            lon_idx = max(0, min(flood_mask.shape[1] - 1, lon_idx))
            lat_idx = max(0, min(flood_mask.shape[0] - 1, lat_idx))

            is_flooded = flood_mask[lat_idx, lon_idx] if lat_idx < flood_mask.shape[0] and lon_idx < flood_mask.shape[1] else False

            # Calculate water depth at sensor
            water_depth = max(0, water_level - sensor['elevation']) if is_flooded else 0

            # Update water level readings
            if is_flooded:
                # Water level reading = baseline + actual water depth + small noise
                sensor['current_reading'] = sensor['baseline_reading'] + water_depth + random.uniform(-0.1, 0.1)
            else:
                # Normal baseline reading with small variations
                sensor['current_reading'] = sensor['baseline_reading'] + random.uniform(-0.05, 0.05)

            # Determine alert level based on water depth
            if is_flooded and water_depth > 2:
                sensor['alert_level'] = 'critical'
                sensor['status'] = 'flooding_detected'
            elif is_flooded and water_depth > 0.5:
                sensor['alert_level'] = 'warning'
                sensor['status'] = 'water_detected'
            elif is_flooded:
                sensor['alert_level'] = 'caution'
                sensor['status'] = 'moisture_detected'
            else:
                sensor['alert_level'] = 'normal'
                sensor['status'] = 'operational'

            sensor['is_flooded'] = is_flooded
            sensor['water_depth'] = water_depth
            sensor['last_update'] = datetime.now()

            # Simulate sensor failure in extreme conditions
            if water_depth > 3 and random.random() < 0.1:
                sensor['status'] = 'offline'
                sensor['current_reading'] = 0

            updated_sensors.append(sensor)

        self.sensors = updated_sensors
        return updated_sensors

    def should_stream_data(self) -> bool:
        """Check if 10 seconds have passed since last stream"""
        current_time = datetime.now()
        time_diff = (current_time - self.last_stream_time).total_seconds()

        if time_diff >= 10:
            self.last_stream_time = current_time
            return True
        return False

    def get_sensor_summary(self) -> Dict:
        """Get summary statistics of sensor network"""
        operational = sum(1 for s in self.sensors if s['status'] != 'offline')
        flooded = sum(1 for s in self.sensors if s['is_flooded'])
        critical = sum(1 for s in self.sensors if s['alert_level'] == 'critical')
        warning = sum(1 for s in self.sensors if s['alert_level'] == 'warning')

        return {
            'total_sensors': len(self.sensors),
            'operational': operational,
            'offline': len(self.sensors) - operational,
            'flooded_sensors': flooded,
            'critical_alerts': critical,
            'warning_alerts': warning,
            'normal': len(self.sensors) - critical - warning
        }

    def add_sensor_data_to_stream(self, redis_client, stream_name="sensor_data"):
        """Add all water level sensor readings to Redis stream every 10 seconds"""
        if not redis_client or not self.should_stream_data():
            return False

        try:
            for sensor in self.sensors:
                sensor_data = {
                    'sensor_id': sensor['id'],
                    'sensor_type': sensor['type'],
                    'lat': str(sensor['lat']),
                    'lon': str(sensor['lon']),
                    'elevation': str(sensor['elevation']),
                    'current_reading': str(sensor['current_reading']),
                    'water_depth': str(sensor.get('water_depth', 0)),
                    'status': sensor['status'],
                    'alert_level': sensor['alert_level'],
                    'is_flooded': str(sensor['is_flooded']),
                    'timestamp': sensor['last_update'].isoformat()
                }

                # Add to Redis stream
                redis_client.xadd(stream_name, sensor_data)

            return True
        except Exception as e:
            st.error(f"Error adding sensor data to stream: {e}")
            return False

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
        self.last_tweet_time = datetime.now()

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
            "Enjoying the sunshine today",
            "Perfect weather for outdoor activities in {city}",
            "Morning jog complete! Feeling great 🏃‍♂️",
            "Coffee shop in {city} has the best wifi",
            "Another productive day at work",
            "Looking forward to the weekend!"
        ]

        self.flood_tweets_mild = [
            "Roads are getting a bit wet in {city} 🌧️",
            "Some puddles forming on the streets",
            "Rain is picking up here in {city}",
            "Storm clouds gathering overhead ⛈️",
            "Getting a bit soggy out there",
            "Weather alert just came in for our area",
            "Streets starting to flood a little",
            "Water levels rising in the creek",
            "Heavy rain causing some street flooding in {city}",
            "Drainage struggling to keep up with the rain"
        ]

        self.flood_tweets_moderate = [
            "Water rising quickly on {street}! 🚨",
            "Basement starting to flood, need help!",
            "Can't get through downtown {city} - roads flooded",
            "Emergency services are overwhelmed",
            "Water up to my car's tires 😰",
            "Evacuation notice just issued for our area",
            "Power went out, water everywhere!",
            "Need rescue at {street} intersection!",
            "Major flooding on {street} in {city}",
            "Emergency crews responding to flood calls"
        ]

        self.flood_tweets_severe = [
            "HELP! Trapped by flood water at {street}! 🆘",
            "Water up to second floor! Need immediate rescue!",
            "EMERGENCY: Family stranded on roof at {street}",
            "Flood waters rising rapidly! Send help NOW!",
            "Can't evacuate - roads completely flooded!",
            "URGENT: Need boat rescue in {city}!",
            "Water everywhere! This is catastrophic! 😱",
            "FLASH FLOOD WARNING! Get out NOW!",
            "SEVERE FLOODING - CALL 911 if trapped!",
            "Historic flood levels in {city} - stay safe!"
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
            "Food trend alert: Everyone's trying this! 🥑",
            "Stock market volatility continues 📊",
            "New restaurant opening downtown!",
            "Concert tickets on sale now! 🎶",
            "Breaking: Local team wins championship!",
            "Weather forecast looks good this week",
            "New shopping mall opens next month",
            "Local festival this weekend! 🎪",
            "Best pizza in town at Mario's!"
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

    def should_generate_tweet(self, tweet_rate: int) -> bool:
        """Check if it's time to generate a new tweet based on the rate"""
        current_time = datetime.now()
        time_diff = (current_time - self.last_tweet_time).total_seconds()

        # Calculate interval: tweets_per_minute to seconds_per_tweet
        interval = 60.0 / tweet_rate if tweet_rate > 0 else 60.0

        if time_diff >= interval:
            self.last_tweet_time = current_time
            return True
        return False

    def generate_continuous_tweets(self, extent: List[float], flood_mask: np.ndarray,
                                   elev_array: np.ndarray, water_level: float,
                                   city_name: str, tweet_rate: int) -> List[Dict]:
        """Generate tweets continuously based on flood conditions"""
        if not self.should_generate_tweet(tweet_rate):
            return []

        tweets = []

        # Determine if there's significant flooding
        flooding_percentage = (np.sum(flood_mask) / flood_mask.size) * 100 if flood_mask.size > 0 else 0
        is_flooding = flooding_percentage > 1.0  # Consider flooding if >1% of area is flooded

        # Generate 1-3 tweets per interval
        num_tweets = random.randint(1, 3)

        for _ in range(num_tweets):
            if is_flooding:
                # During flooding: 60% flood-related, 40% noise
                is_flood_related = random.random() < 0.6

                if is_flood_related:
                    # 70% genuine flood tweets, 30% noise users posting about flood
                    is_genuine = random.random() < 0.7
                    tweet = self._generate_flood_tweet(extent, flood_mask, elev_array,
                                                       water_level, city_name, is_genuine)
                else:
                    # Noise tweets during flooding
                    tweet = self._generate_noise_tweet(extent, city_name)
            else:
                # No flooding: 90% normal/noise, 10% scattered flood mentions
                is_flood_mention = random.random() < 0.1

                if is_flood_mention:
                    # Occasional false alarm or old flood reference
                    tweet = self._generate_mild_flood_tweet(extent, city_name, is_genuine=False)
                else:
                    # Mix of normal tweets and noise
                    is_normal = random.random() < 0.5
                    if is_normal:
                        tweet = self._generate_normal_tweet(extent, city_name)
                    else:
                        tweet = self._generate_noise_tweet(extent, city_name)

            tweets.append(tweet)

        return tweets

    def _generate_flood_tweet(self, extent: List[float], flood_mask: np.ndarray,
                              elev_array: np.ndarray, water_level: float,
                              city_name: str, is_genuine: bool) -> Dict:
        """Generate a flood-related tweet"""
        user_pool = self.genuine_users if is_genuine else self.noise_users
        user = random.choice(user_pool)

        # For flood tweets, bias towards flooded areas
        if np.any(flood_mask) and random.random() < 0.8:
            flooded_indices = np.where(flood_mask)
            if len(flooded_indices[0]) > 0:
                idx = random.randint(0, len(flooded_indices[0]) - 1)
                lat_idx, lon_idx = flooded_indices[0][idx], flooded_indices[1][idx]

                # Convert back to lat/lon
                lat = extent[3] - (lat_idx / flood_mask.shape[0]) * (extent[3] - extent[2])
                lon = extent[0] + (lon_idx / flood_mask.shape[1]) * (extent[1] - extent[0])
            else:
                lat = random.uniform(extent[2], extent[3])
                lon = random.uniform(extent[0], extent[1])
        else:
            lat = random.uniform(extent[2], extent[3])
            lon = random.uniform(extent[0], extent[1])

        # Calculate severity
        severity = self._get_flood_severity(lat, lon, extent, flood_mask, elev_array, water_level)

        # Select appropriate template
        if severity <= 0.4:
            template = random.choice(self.flood_tweets_mild)
        elif severity <= 0.7:
            template = random.choice(self.flood_tweets_moderate)
        else:
            template = random.choice(self.flood_tweets_severe)

        streets = ["Main St", "Oak Ave", "First St", "Park Rd", "River Dr", "Hill St", "Center Ave", "Church St"]
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

    def _generate_normal_tweet(self, extent: List[float], city_name: str) -> Dict:
        """Generate a normal non-flood tweet"""
        user = random.choice(self.genuine_users)

        lat = random.uniform(extent[2], extent[3])
        lon = random.uniform(extent[0], extent[1])

        template = random.choice(self.normal_tweets)
        tweet_text = template.format(city=city_name)

        return {
            'user_id': user['user_id'],
            'username': user['username'],
            'lat': lat,
            'lon': lon,
            'text': tweet_text,
            'timestamp': datetime.now().isoformat(),
            'is_genuine': True,
            'flood_severity': 0.0
        }

    def _generate_noise_tweet(self, extent: List[float], city_name: str) -> Dict:
        """Generate a noise tweet"""
        user = random.choice(self.noise_users)

        lat = random.uniform(extent[2], extent[3])
        lon = random.uniform(extent[0], extent[1])

        template = random.choice(self.noise_tweets)
        tweet_text = template.format(city=city_name) if '{city}' in template else template

        return {
            'user_id': user['user_id'],
            'username': user['username'],
            'lat': lat,
            'lon': lon,
            'text': tweet_text,
            'timestamp': datetime.now().isoformat(),
            'is_genuine': False,
            'flood_severity': 0.0
        }

    def _generate_mild_flood_tweet(self, extent: List[float], city_name: str, is_genuine: bool) -> Dict:
        """Generate a mild flood reference tweet"""
        user_pool = self.genuine_users if is_genuine else self.noise_users
        user = random.choice(user_pool)

        lat = random.uniform(extent[2], extent[3])
        lon = random.uniform(extent[0], extent[1])

        template = random.choice(self.flood_tweets_mild)
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
            'flood_severity': 0.3
        }

    def add_tweet_to_stream(self, tweet: Dict):
        """Add tweet to Redis stream only (no local storage)"""
        if self.redis_connected:
            try:
                tweet_data = {k: str(v) for k, v in tweet.items()}
                message_id = self.redis_client.xadd(self.stream_name, tweet_data)
                return message_id
            except Exception as e:
                st.error(f"Redis error: {e}")
                return False
        else:
            # If Redis not connected, silently drop tweets (no local storage)
            return False

    def get_stream_stats(self) -> Dict:
        """Get Redis stream statistics without reading tweet content"""
        if self.redis_connected:
            try:
                stream_info = self.redis_client.xinfo_stream(self.stream_name)
                return {
                    'total_messages': stream_info['length'],
                    'first_entry_id': stream_info.get('first-entry', 'N/A'),
                    'last_entry_id': stream_info.get('last-generated-id', 'N/A'),
                    'connected': True
                }
            except Exception:
                return {
                    'total_messages': 0,
                    'first_entry_id': 'N/A',
                    'last_entry_id': 'N/A',
                    'connected': False
                }
        else:
            return {
                'total_messages': 0,
                'first_entry_id': 'N/A',
                'last_entry_id': 'N/A',
                'connected': False
            }

    def clear_stream(self):
        """Clear all tweets from Redis stream"""
        if self.redis_connected:
            try:
                self.redis_client.delete(self.stream_name)
                return True
            except Exception:
                return False
        else:
            return False

# -------------------------------
# Streamlit Setup
st.set_page_config(layout="wide")
st.title("🌊 Interactive Flood Simulation with Real-Time Data Streaming")

# Initialize tweet generator, sensor network, and flood manager
if 'tweet_generator' not in st.session_state:
    st.session_state.tweet_generator = FloodTweetGenerator()

if 'sensor_network' not in st.session_state:
    st.session_state.sensor_network = FloodSensorNetwork()

if 'flood_manager' not in st.session_state:
    st.session_state.flood_manager = LocalFloodManager()

# -------------------------------
# User Inputs
st.sidebar.header("🌊 Flood Simulation Settings")
place = st.sidebar.text_input("Enter US Location", "Dallas, Texas, USA")
water_level = st.sidebar.slider("Global Water Level (m above sea level)", 0, 300, 100)

# Map display settings
st.sidebar.header("🗺️ Map Display")
use_interactive_map = st.sidebar.checkbox("Enable Interactive Map", True)

st.sidebar.header("📡 Sensor Network Settings")
show_sensors = st.sidebar.checkbox("Show Water Level Sensors", True)
num_sensors = st.sidebar.slider("Number of Water Level Sensors", 5, 50, 20)
sensor_size = st.sidebar.slider("Sensor Icon Size", 50, 300, 100)

st.sidebar.header("🐦 Tweet Stream Settings")
tweet_enabled = st.sidebar.checkbox("Enable Continuous Tweet Streaming", True)
tweet_rate = st.sidebar.slider("Tweets per minute", 5, 120, 30)

if st.sidebar.button("🗑️ Clear Tweet Stream"):
    st.session_state.tweet_generator.clear_stream()
    st.sidebar.success("Tweet stream cleared!")

# -------------------------------
# Get bounding box
try:
    place_gdf = ox.geocode_to_gdf(place)
    place_geom = place_gdf.geometry[0]
    bounds = place_geom.bounds

    st.success(f"📍 Location: {place}")

    # Store in session state
    st.session_state.place = place
    st.session_state.bounds = bounds

except Exception as e:
    st.error(f"Location not found: {e}")
    st.stop()

# -------------------------------
# Fetch elevation data
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

        # Store in session state
        st.session_state.elev_array = elev_array
        st.session_state.extent = extent
        st.session_state.water_level = water_level

        st.info(f"Elevation range: {np.nanmin(elev_array):.1f}m to {np.nanmax(elev_array):.1f}m")

    except Exception as e:
        st.error(f"Critical error: {e}")
        st.stop()

# -------------------------------
# Flood Simulation (global only - no local floods)
global_flood_mask = elev_array <= water_level
flooded_area_pct = (np.sum(global_flood_mask) / global_flood_mask.size) * 100

# Store flood mask in session state
st.session_state.flood_mask = global_flood_mask

st.info(f"💧 Flooded area: {flooded_area_pct:.1f}% of the region")

# -------------------------------
# Deploy and update sensors
if show_sensors:
    # Deploy sensors if not already deployed or if number changed
    if not hasattr(st.session_state.sensor_network, 'sensors') or len(st.session_state.sensor_network.sensors) != num_sensors:
        st.session_state.sensor_network.num_sensors = num_sensors
        sensors = st.session_state.sensor_network.deploy_sensors(extent, elev_array)
        st.success(f"📡 Deployed {len(sensors)} water level sensors across the area")

    # Update sensor readings based on current flood conditions
    sensors = st.session_state.sensor_network.update_sensor_readings(extent, global_flood_mask, elev_array, water_level)
    sensor_summary = st.session_state.sensor_network.get_sensor_summary()

    # Stream sensor data to Redis every 10 seconds
    if st.session_state.tweet_generator.redis_connected:
        success = st.session_state.sensor_network.add_sensor_data_to_stream(
            st.session_state.tweet_generator.redis_client,
            "sensor_data"
        )

else:
    sensors = []
    sensor_summary = {}

# -------------------------------
# Create layout with interactive map and minimal status
col1, col2 = st.columns([4, 1])

with col1:
    st.subheader("🗺️ Flood Simulation Map with Sensor Network")

    if use_interactive_map:
        # Create interactive Plotly map
        with st.spinner("Creating interactive map..."):
            try:
                fig = go.Figure()

                # Create custom terrain-like colorscale
                terrain_colorscale = [
                    [0.0, '#0066cc'],    [0.1, '#004499'],    [0.2, '#66cc99'],
                    [0.4, '#99cc66'],    [0.6, '#cccc33'],    [0.8, '#cc9933'],
                    [1.0, '#996633']
                ]

                # Add elevation heatmap
                fig.add_trace(go.Heatmap(
                    z=elev_array,
                    x=np.linspace(extent[0], extent[1], elev_array.shape[1]),
                    y=np.linspace(extent[2], extent[3], elev_array.shape[0]),
                    colorscale=terrain_colorscale,
                    opacity=0.8,
                    name='Elevation',
                    colorbar=dict(title="Elevation (m)", x=1.02)
                ))

                # Add flood overlay if there's flooding
                if np.any(global_flood_mask):
                    flood_overlay = np.where(global_flood_mask, 1, np.nan)
                    fig.add_trace(go.Heatmap(
                        z=flood_overlay,
                        x=np.linspace(extent[0], extent[1], elev_array.shape[1]),
                        y=np.linspace(extent[2], extent[3], elev_array.shape[0]),
                        colorscale=[[0, 'rgba(0,0,255,0)'], [1, 'rgba(0,100,255,0.6)']],
                        showscale=False,
                        name='Flood Area',
                        hovertemplate='Flooded Area<extra></extra>'
                    ))

                # Add sensors if enabled
                if show_sensors and sensors:
                    sensor_colors = {
                        'normal': 'green',
                        'caution': 'yellow',
                        'warning': 'orange',
                        'critical': 'red'
                    }

                    for alert_level, color in sensor_colors.items():
                        level_sensors = [s for s in sensors if s['alert_level'] == alert_level]
                        if level_sensors:
                            fig.add_trace(go.Scatter(
                                x=[s['lon'] for s in level_sensors],
                                y=[s['lat'] for s in level_sensors],
                                mode='markers',
                                marker=dict(
                                    size=sensor_size/10,
                                    color=color,
                                    line=dict(width=2, color='black'),
                                    symbol='circle' if alert_level == 'normal' else
                                    'square' if alert_level == 'caution' else
                                    'triangle-up' if alert_level == 'warning' else 'x'
                                ),
                                name=f'{alert_level.title()} Water Level Sensors ({len(level_sensors)})',
                                text=[f"{s['id']}<br>Type: {s['type']}<br>Reading: {s['current_reading']:.2f}<br>Status: {s['status']}"
                                      for s in level_sensors],
                                hovertemplate='%{text}<extra></extra>'
                            ))

                # Configure layout
                fig.update_layout(
                    title=f"Flood Map: {place}<br>Water Level: {water_level}m | Flooded Area: {flooded_area_pct:.1f}%",
                    xaxis_title="Longitude (°)",
                    yaxis_title="Latitude (°)",
                    height=700,
                    showlegend=True,
                    legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
                    hovermode='closest'
                )

                # Equal aspect ratio
                fig.update_yaxes(scaleanchor="x", scaleratio=1)

                # Display the interactive map (read-only)
                st.plotly_chart(fig, use_container_width=True, key="flood_map")

            except Exception as e:
                st.error(f"Interactive map error: {e}")
                use_interactive_map = False

    # Fallback static map (matplotlib)
    if not use_interactive_map:
        with st.spinner("Creating static map..."):
            try:
                fig, ax = plt.subplots(figsize=(12, 10), dpi=100)

                # Plot elevation
                elev_img = ax.imshow(elev_array, cmap='terrain', extent=extent, origin='upper', alpha=0.9)
                fig.colorbar(elev_img, ax=ax, label="Elevation (m)", shrink=0.8)

                # Plot flood overlay
                if np.any(global_flood_mask):
                    flood_overlay = np.ma.masked_where(~global_flood_mask, np.ones_like(global_flood_mask))
                    ax.imshow(flood_overlay, cmap='Blues', alpha=0.6, extent=extent, origin='upper')

                # Plot sensors if enabled
                if show_sensors and sensors:
                    sensor_groups = {
                        'normal': [],
                        'caution': [],
                        'warning': [],
                        'critical': []
                    }

                    for sensor in sensors:
                        sensor_groups[sensor['alert_level']].append(sensor)

                    group_styles = {
                        'normal': {'color': 'green', 'marker': 'o', 'size_mult': 0.8, 'label': 'Normal'},
                        'caution': {'color': 'yellow', 'marker': 's', 'size_mult': 1.0, 'label': 'Caution'},
                        'warning': {'color': 'orange', 'marker': '^', 'size_mult': 1.2, 'label': 'Warning'},
                        'critical': {'color': 'red', 'marker': 'X', 'size_mult': 1.5, 'label': 'Critical'}
                    }

                    for alert_level, group_sensors in sensor_groups.items():
                        if group_sensors:
                            style = group_styles[alert_level]
                            lats = [s['lat'] for s in group_sensors]
                            lons = [s['lon'] for s in group_sensors]

                            ax.scatter(lons, lats,
                                       c=style['color'],
                                       marker=style['marker'],
                                       s=sensor_size * style['size_mult'],
                                       alpha=0.9,
                                       edgecolors='black',
                                       linewidth=2,
                                       label=f"{style['label']} ({len(group_sensors)})",
                                       zorder=10)

                ax.set_xlabel("Longitude (°)")
                ax.set_ylabel("Latitude (°)")
                ax.set_title(f"Flood Simulation: {place}\nWater Level: {water_level}m | Flooded Area: {flooded_area_pct:.1f}%")
                ax.grid(True, alpha=0.3)

                if show_sensors and sensors:
                    ax.legend(loc='upper left', title='Sensor Status', frameon=True, fancybox=True, shadow=True)

                st.pyplot(fig, clear_figure=True, use_container_width=True)
                plt.close(fig)

            except Exception as e:
                st.error(f"Map rendering error: {e}")

with col2:
    st.subheader("📊 Streaming Status")

    # Redis connection status
    if st.session_state.tweet_generator.redis_connected:
        st.success("🟢 Redis Connected")
    else:
        st.error("🔴 Redis Disconnected")
        st.caption("Data streams disabled")

    st.write("---")

    # Sensor statistics
    if show_sensors and sensor_summary:
        st.subheader("📡 Water Level Sensors")
        st.metric("Total Sensors", sensor_summary['total_sensors'])
        st.metric("Critical Alerts", sensor_summary['critical_alerts'])
        st.metric("Warning Alerts", sensor_summary['warning_alerts'])
        st.metric("Operational", sensor_summary['operational'])

    st.write("---")

    # Tweet stream statistics (without showing content)
    if tweet_enabled:
        st.subheader("🐦 Continuous Tweet Stream")
        tweet_stats = st.session_state.tweet_generator.get_stream_stats()

        if tweet_stats['connected']:
            st.metric("Total Tweets", tweet_stats['total_messages'])
            st.metric("Rate (per min)", tweet_rate)
            flooding_pct = (np.sum(global_flood_mask) / global_flood_mask.size) * 100
            if flooding_pct > 1:
                st.caption("🌊 Flood mode: More flood tweets")
            else:
                st.caption("☀️ Normal mode: Mixed content")
        else:
            st.error("Stream not available")

    # Manual refresh button
    if st.button("🔄 Force Update", type="primary"):
        if show_sensors and st.session_state.tweet_generator.redis_connected:
            st.session_state.sensor_network.add_sensor_data_to_stream(
                st.session_state.tweet_generator.redis_client,
                "sensor_data"
            )
        st.rerun()

# -------------------------------
# Background data generation - Continuous streaming
if st.session_state.tweet_generator.redis_connected:
    city_name = place.split(',')[0]

    # Generate continuous tweets
    if tweet_enabled:
        new_tweets = st.session_state.tweet_generator.generate_continuous_tweets(
            extent, global_flood_mask, elev_array, water_level, city_name, tweet_rate
        )

        # Stream all generated tweets to Redis
        for tweet in new_tweets:
            st.session_state.tweet_generator.add_tweet_to_stream(tweet)

# -------------------------------
# Bottom statistics dashboard
st.subheader("📊 Simulation Dashboard")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Global Water Level", f"{water_level} m")
with col2:
    st.metric("Total Flooded Area", f"{flooded_area_pct:.1f}%")
with col3:
    st.metric("Elevation Range", f"{np.nanmin(elev_array):.0f}m - {np.nanmax(elev_array):.0f}m")
with col4:
    if tweet_enabled and st.session_state.tweet_generator.redis_connected:
        tweet_stats = st.session_state.tweet_generator.get_stream_stats()
        st.metric("Tweets Streamed", tweet_stats['total_messages'])
    else:
        st.metric("Tweets Streamed", "Disabled")
with col5:
    if show_sensors and sensor_summary:
        st.metric("Active Sensors", sensor_summary['operational'])
    else:
        st.metric("Active Sensors", "0")
with col6:
    st.metric("Water Level Sensors", f"{len(sensors) if sensors else 0} sensors")

# Detailed sensor data table (if sensors enabled)
if show_sensors and sensors:
    st.subheader("📋 Water Level Sensor Network Details")

    # Create sensor DataFrame
    sensor_data = []
    for sensor in sensors:
        sensor_data.append({
            'ID': sensor['id'],
            'Type': 'Water Level',
            'Location': f"{sensor['lat']:.4f}, {sensor['lon']:.4f}",
            'Elevation': f"{sensor['elevation']:.1f}m",
            'Water Level Reading': f"{sensor['current_reading']:.2f}m",
            'Water Depth': f"{sensor.get('water_depth', 0):.1f}m",
            'Alert Level': sensor['alert_level'],
            'Status': sensor['status'],
            'Last Update': sensor['last_update'].strftime('%H:%M:%S')
        })

    df = pd.DataFrame(sensor_data)

    # Color-code the table based on alert level
    def color_alert_level(val):
        if val == 'critical':
            return 'background-color: #ffcccc'
        elif val == 'warning':
            return 'background-color: #fff3cd'
        elif val == 'caution':
            return 'background-color: #d1ecf1'
        else:
            return 'background-color: #d4edda'

    styled_df = df.style.applymap(color_alert_level, subset=['Alert Level'])
    st.dataframe(styled_df, use_container_width=True, height=300)

st.info("📡 **Sensor Data**: Water level sensors stream data to Redis every 10 seconds")
st.info("🐦 **Tweet Data**: Continuous tweet generation based on flood conditions and city activity")

st.warning("⚠️ This is a simplified flood simulation for educational purposes only. Real flood modeling requires additional factors like rainfall, drainage, soil permeability, and temporal dynamics.")

# Auto-refresh mechanism for continuous data streaming
if tweet_enabled or show_sensors:
    # Initialize auto-refresh counter
    if 'auto_refresh_counter' not in st.session_state:
        st.session_state.auto_refresh_counter = 0

    # Auto-refresh to maintain continuous streaming
    auto_refresh = st.sidebar.checkbox("Auto-refresh for continuous streaming", value=True)
    if auto_refresh:
        time.sleep(1)  # Fast refresh for continuous streaming
        st.rerun()