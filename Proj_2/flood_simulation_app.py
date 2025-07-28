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
        """Initialize flood sensor network"""
        self.sensors = []
        self.num_sensors = num_sensors

    def deploy_sensors(self, extent: List[float], elev_array: np.ndarray) -> List[Dict]:
        """Deploy sensors randomly across the map area"""
        sensors = []

        # Sensor types and their characteristics
        sensor_types = {
            'water_level': {'icon': '🌊', 'color': 'blue', 'baseline': 0.1},
            'rain_gauge': {'icon': '🌧️', 'color': 'green', 'baseline': 0.0},
            'flow_meter': {'icon': '📏', 'color': 'purple', 'baseline': 0.2},
            'pressure': {'icon': '⚡', 'color': 'orange', 'baseline': 0.05}
        }

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

            # Select random sensor type
            sensor_type = random.choice(list(sensor_types.keys()))
            sensor_info = sensor_types[sensor_type]

            sensor = {
                'id': f'SENSOR_{i+1:03d}',
                'type': sensor_type,
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
        """Update sensor readings based on current flood conditions"""
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

            # Update readings based on sensor type and flood conditions
            if sensor['type'] == 'water_level':
                if is_flooded:
                    sensor['current_reading'] = sensor['baseline_reading'] + water_depth + random.uniform(-0.1, 0.1)
                else:
                    sensor['current_reading'] = sensor['baseline_reading'] + random.uniform(-0.05, 0.05)

            elif sensor['type'] == 'rain_gauge':
                if is_flooded:
                    sensor['current_reading'] = random.uniform(50, 150)  # Heavy rain mm/hr
                else:
                    sensor['current_reading'] = random.uniform(0, 10)  # Light/no rain

            elif sensor['type'] == 'flow_meter':
                if is_flooded:
                    sensor['current_reading'] = sensor['baseline_reading'] + water_depth * 2 + random.uniform(0, 1)
                else:
                    sensor['current_reading'] = sensor['baseline_reading'] + random.uniform(-0.1, 0.1)

            elif sensor['type'] == 'pressure':
                if is_flooded:
                    sensor['current_reading'] = sensor['baseline_reading'] + water_depth * 0.1 + random.uniform(0, 0.02)
                else:
                    sensor['current_reading'] = sensor['baseline_reading'] + random.uniform(-0.01, 0.01)

            # Determine alert level
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
        """Add all sensor readings to Redis stream"""
        if not redis_client:
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
show_rivers = st.sidebar.checkbox("Overlay Rivers", True)
show_buildings = st.sidebar.checkbox("Overlay Buildings", False)

# Interactive flooding controls
st.sidebar.header("🎯 Interactive Flooding")
use_interactive_map = st.sidebar.checkbox("Enable Interactive Map", True)
if st.sidebar.button("🗑️ Clear All Local Floods"):
    st.session_state.flood_manager.clear_all_floods()
    st.sidebar.success("Local floods cleared!")

# Show active flood events
if st.session_state.flood_manager.flood_events:
    st.sidebar.write("**Active Local Floods:**")
    for event in st.session_state.flood_manager.flood_events:
        col_a, col_b = st.sidebar.columns([3, 1])
        with col_a:
            st.sidebar.caption(f"{event['id']}: {event['water_level']}m at ({event['lat']:.3f}, {event['lon']:.3f})")
        with col_b:
            if st.sidebar.button("❌", key=f"remove_{event['id']}"):
                st.session_state.flood_manager.remove_flood_event(event['id'])
                st.rerun()

st.sidebar.header("📡 Sensor Network Settings")
show_sensors = st.sidebar.checkbox("Show Sensors", True)
num_sensors = st.sidebar.slider("Number of Sensors", 5, 50, 20)
sensor_size = st.sidebar.slider("Sensor Icon Size", 50, 300, 100)
stream_sensors = st.sidebar.checkbox("Stream Sensor Data to Redis", True)
sensor_stream_interval = st.sidebar.slider("Sensor Stream Interval (seconds)", 5, 60, 10)

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

        # Store in session state for tweet generation
        st.session_state.elev_array = elev_array
        st.session_state.extent = extent
        st.session_state.water_level = water_level

        st.info(f"Elevation range: {np.nanmin(elev_array):.1f}m to {np.nanmax(elev_array):.1f}m")

    except Exception as e:
        st.error(f"Critical error: {e}")
        st.stop()

# -------------------------------
# Flood Simulation (with local floods)
# Use combined flood mask that includes both global and local flooding
combined_flood_mask = st.session_state.flood_manager.calculate_combined_flood_mask(
    extent, elev_array, water_level
)
flooded_area_pct = (np.sum(combined_flood_mask) / combined_flood_mask.size) * 100

# Store both masks in session state
st.session_state.flood_mask = combined_flood_mask
st.session_state.global_flood_mask = elev_array <= water_level

st.info(f"💧 Total flooded area: {flooded_area_pct:.1f}% of the region")
if st.session_state.flood_manager.flood_events:
    st.info(f"🎯 Active local flood events: {len(st.session_state.flood_manager.flood_events)}")

# -------------------------------
# Deploy and update sensors
if show_sensors:
    # Deploy sensors if not already deployed or if number changed
    if not hasattr(st.session_state.sensor_network, 'sensors') or len(st.session_state.sensor_network.sensors) != num_sensors:
        st.session_state.sensor_network.num_sensors = num_sensors
        sensors = st.session_state.sensor_network.deploy_sensors(extent, elev_array)
        st.success(f"📡 Deployed {len(sensors)} sensors across the area")

    # Update sensor readings based on current flood conditions (use combined mask)
    sensors = st.session_state.sensor_network.update_sensor_readings(extent, combined_flood_mask, elev_array, water_level)
    sensor_summary = st.session_state.sensor_network.get_sensor_summary()

    # Start sensor data streaming to Redis if enabled
    if stream_sensors and st.session_state.tweet_generator.redis_connected:
        # Initialize streaming counter
        if 'sensor_stream_counter' not in st.session_state:
            st.session_state.sensor_stream_counter = 0

        # Stream sensor data every interval
        st.session_state.sensor_stream_counter += 1
        if st.session_state.sensor_stream_counter % max(1, sensor_stream_interval // 3) == 0:
            success = st.session_state.sensor_network.add_sensor_data_to_stream(
                st.session_state.tweet_generator.redis_client,
                "sensor_data"
            )
            if success:
                st.sidebar.success(f"📡 Sensor data streamed at {datetime.now().strftime('%H:%M:%S')}")

else:
    sensors = []
    sensor_summary = {}

# -------------------------------
# Create layout with interactive map, sensors, and tweets
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    st.subheader("🗺️ Interactive Flood Simulation Map")

    if use_interactive_map:
        # Create interactive Plotly map
        with st.spinner("Creating interactive map..."):
            try:
                # Create the base elevation map
                fig = go.Figure()

                # Create custom terrain-like colorscale
                terrain_colorscale = [
                    [0.0, '#0066cc'],    # Deep water (blue)
                    [0.1, '#004499'],    # Shallow water (dark blue)
                    [0.2, '#66cc99'],    # Low elevation (green)
                    [0.4, '#99cc66'],    # Medium-low elevation (light green)
                    [0.6, '#cccc33'],    # Medium elevation (yellow)
                    [0.8, '#cc9933'],    # High elevation (orange)
                    [1.0, '#996633']     # Very high elevation (brown)
                ]

                # Add elevation heatmap
                fig.add_trace(go.Heatmap(
                    z=elev_array,
                    x=np.linspace(extent[0], extent[1], elev_array.shape[1]),
                    y=np.linspace(extent[2], extent[3], elev_array.shape[0]),
                    colorscale=terrain_colorscale,  # Use custom terrain colorscale
                    opacity=0.8,
                    name='Elevation',
                    colorbar=dict(title="Elevation (m)", x=1.02)
                ))

                # Add flood overlay if there's flooding
                if np.any(combined_flood_mask):
                    # Create flood overlay
                    flood_overlay = np.where(combined_flood_mask, 1, np.nan)
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
                                name=f'{alert_level.title()} Sensors ({len(level_sensors)})',
                                text=[f"{s['id']}<br>Type: {s['type']}<br>Reading: {s['current_reading']:.2f}<br>Status: {s['status']}"
                                      for s in level_sensors],
                                hovertemplate='%{text}<extra></extra>'
                            ))

                # Add local flood event markers
                if st.session_state.flood_manager.flood_events:
                    flood_events = st.session_state.flood_manager.flood_events
                    fig.add_trace(go.Scatter(
                        x=[event['lon'] for event in flood_events],
                        y=[event['lat'] for event in flood_events],
                        mode='markers',
                        marker=dict(
                            size=20,
                            color='red',
                            symbol='circle-open',
                            line=dict(width=4, color='red')
                        ),
                        name='Local Flood Events',
                        text=[f"{event['id']}<br>Water Level: {event['water_level']}m<br>Radius: {event['radius_km']}km"
                              for event in flood_events],
                        hovertemplate='%{text}<extra></extra>'
                    ))

                # Configure layout
                fig.update_layout(
                    title=f"Interactive Flood Map: {place}<br>Global Water Level: {water_level}m | Flooded Area: {flooded_area_pct:.1f}%",
                    xaxis_title="Longitude (°)",
                    yaxis_title="Latitude (°)",
                    height=600,
                    showlegend=True,
                    legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
                    hovermode='closest'
                )

                # Equal aspect ratio
                fig.update_yaxes(scaleanchor="x", scaleratio=1)

                # Display the interactive map
                selected_points = st.plotly_chart(fig, use_container_width=True, key="flood_map")

                # Instructions for interaction
                st.info("💡 **Instructions**: The map above shows real-time flood conditions. Use the controls below to add localized flooding.")

                # Local flood creation interface
                st.write("### 🎯 Create Local Flood Event")
                col_flood1, col_flood2, col_flood3 = st.columns(3)

                with col_flood1:
                    local_lat = st.number_input("Latitude",
                                                min_value=extent[2],
                                                max_value=extent[3],
                                                value=(extent[2] + extent[3])/2,
                                                step=0.001,
                                                format="%.6f")
                with col_flood2:
                    local_lon = st.number_input("Longitude",
                                                min_value=extent[0],
                                                max_value=extent[1],
                                                value=(extent[0] + extent[1])/2,
                                                step=0.001,
                                                format="%.6f")
                with col_flood3:
                    local_water_level = st.number_input("Local Water Level (m)",
                                                        min_value=0,
                                                        max_value=500,
                                                        value=water_level + 10,
                                                        step=1)

                col_flood4, col_flood5 = st.columns(2)
                with col_flood4:
                    flood_radius = st.slider("Flood Radius (km)", 0.5, 10.0, 2.0, 0.5)
                with col_flood5:
                    if st.button("🌊 Create Local Flood", type="primary"):
                        event = st.session_state.flood_manager.add_flood_event(
                            local_lat, local_lon, local_water_level, flood_radius
                        )
                        st.success(f"Created flood event {event['id']} at ({local_lat:.3f}, {local_lon:.3f})")
                        st.rerun()

            except Exception as e:
                st.error(f"Interactive map error: {e}")
                # Fallback to matplotlib
                st.info("Falling back to static map...")
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
                if np.any(combined_flood_mask):
                    flood_overlay = np.ma.masked_where(~combined_flood_mask, np.ones_like(combined_flood_mask))
                    ax.imshow(flood_overlay, cmap='Blues', alpha=0.6, extent=extent, origin='upper')

                # Plot sensors on the map with enhanced visibility
                if show_sensors and sensors:
                    # Group sensors by alert level for better plotting
                    sensor_groups = {
                        'normal': [],
                        'caution': [],
                        'warning': [],
                        'critical': []
                    }

                    for sensor in sensors:
                        sensor_groups[sensor['alert_level']].append(sensor)

                    # Plot each group with different styles
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
                                       zorder=10)  # Ensure sensors appear on top

                    # Add sensor IDs as text labels for critical sensors
                    critical_sensors = [s for s in sensors if s['alert_level'] == 'critical']
                    for sensor in critical_sensors:
                        ax.annotate(sensor['id'][-3:],
                                    (sensor['lon'], sensor['lat']),
                                    xytext=(5, 5),
                                    textcoords='offset points',
                                    fontsize=8,
                                    color='white',
                                    weight='bold',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7))

                # Plot local flood events
                if st.session_state.flood_manager.flood_events:
                    for event in st.session_state.flood_manager.flood_events:
                        # Draw flood event center
                        ax.scatter(event['lon'], event['lat'],
                                   s=300, c='red', marker='*',
                                   edgecolors='darkred', linewidth=3,
                                   label='Local Floods' if event == st.session_state.flood_manager.flood_events[0] else "",
                                   zorder=15)

                        # Draw radius circle
                        radius_deg = event['radius_km'] / 111.0
                        circle = plt.Circle((event['lon'], event['lat']), radius_deg,
                                            fill=False, color='red', linewidth=2, linestyle='--', alpha=0.7)
                        ax.add_patch(circle)

                        # Add label
                        ax.annotate(f"{event['id']}\n{event['water_level']}m",
                                    (event['lon'], event['lat']),
                                    xytext=(10, 10),
                                    textcoords='offset points',
                                    fontsize=9,
                                    color='darkred',
                                    weight='bold',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

                # Add coordinate system
                ax.set_xlabel("Longitude (°)")
                ax.set_ylabel("Latitude (°)")
                ax.set_title(f"Flood Simulation: {place}\nGlobal Level: {water_level}m | Total Flooded: {flooded_area_pct:.1f}%")
                ax.grid(True, alpha=0.3)

                # Add legend for sensors if shown
                if show_sensors and sensors:
                    ax.legend(loc='upper left', title='Sensor Status', frameon=True, fancybox=True, shadow=True)

                st.pyplot(fig, clear_figure=True, use_container_width=True)
                plt.close(fig)

            except Exception as e:
                st.error(f"Map rendering error: {e}")

with col2:
    st.subheader("📡 Sensor Network Status")

    if show_sensors and sensor_summary:
        # Show streaming status
        if stream_sensors:
            if st.session_state.tweet_generator.redis_connected:
                st.success("🔴 LIVE - Streaming to Redis")
                st.caption(f"📡 Interval: {sensor_stream_interval}s")
            else:
                st.warning("⚠️ Redis not connected")
        else:
            st.info("📡 Streaming disabled")

        # Sensor network overview
        st.metric("Total Sensors", sensor_summary['total_sensors'])

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Operational", sensor_summary['operational'])
            st.metric("Critical", sensor_summary['critical_alerts'])
        with col_s2:
            st.metric("Offline", sensor_summary['offline'])
            st.metric("Warning", sensor_summary['warning_alerts'])

        st.write("---")

        # Show Redis stream info if connected
        if stream_sensors and st.session_state.tweet_generator.redis_connected:
            try:
                stream_info = st.session_state.tweet_generator.redis_client.xinfo_stream("sensor_data")
                st.caption(f"📊 Stream length: {stream_info['length']} messages")
            except:
                st.caption("📊 Starting sensor stream...")

        # Live sensor readings
        st.subheader("🔴 Critical Alerts")
        critical_sensors = [s for s in sensors if s['alert_level'] == 'critical']
        if critical_sensors:
            for sensor in critical_sensors:
                st.error(f"🚨 **{sensor['id']}** ({sensor['type']})\n"
                         f"Reading: {sensor['current_reading']:.2f}\n"
                         f"Depth: {sensor.get('water_depth', 0):.1f}m")
        else:
            st.success("No critical alerts")

        st.subheader("🟡 Warning Sensors")
        warning_sensors = [s for s in sensors if s['alert_level'] == 'warning']
        if warning_sensors:
            for sensor in warning_sensors[:3]:  # Show top 3
                st.warning(f"⚠️ **{sensor['id']}** ({sensor['type']})\n"
                           f"Reading: {sensor['current_reading']:.2f}\n"
                           f"Depth: {sensor.get('water_depth', 0):.1f}m")
        else:
            st.info("No warnings")

        # Refresh button for sensors
        if st.button("🔄 Update Sensors", key="sensor_refresh"):
            st.session_state.sensor_network.update_sensor_readings(extent, combined_flood_mask, elev_array, water_level)
            # Force a sensor data stream update
            if stream_sensors and st.session_state.tweet_generator.redis_connected:
                st.session_state.sensor_network.add_sensor_data_to_stream(
                    st.session_state.tweet_generator.redis_client,
                    "sensor_data"
                )
                st.success("📡 Sensor data streamed!")
            st.rerun()

    else:
        st.info("Sensor network disabled. Enable in sidebar to see sensor data.")

with col3:
    st.subheader("🐦 Live Tweet Feed")

    # Generate tweets if enabled (use combined flood mask)
    if tweet_enabled:
        city_name = place.split(',')[0]
        for _ in range(random.randint(1, 2)):
            tweet = st.session_state.tweet_generator.generate_tweet(
                extent, combined_flood_mask, elev_array, water_level, city_name
            )
            st.session_state.tweet_generator.add_tweet_to_stream(tweet)

    if tweet_enabled:
        # Tweet stream status
        if st.session_state.tweet_generator.redis_connected:
            st.success("🟢 Redis Connected")
        else:
            st.info("🟡 Using Local Storage")

        # Get and display recent tweets
        recent_tweets = st.session_state.tweet_generator.get_recent_tweets(10)

        if recent_tweets:
            # Calculate statistics
            genuine_count = sum(1 for t in recent_tweets if t['is_genuine'])
            noise_count = len(recent_tweets) - genuine_count

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Genuine", genuine_count)
            with col_b:
                st.metric("Noise", noise_count)

            st.write("---")

            # Display tweets with color coding (compact version)
            for tweet in reversed(recent_tweets[-8:]):  # Show latest 8
                timestamp = tweet['timestamp'][:16] if isinstance(tweet['timestamp'], str) else str(tweet['timestamp'])[:16]

                if tweet['is_genuine']:
                    if tweet['flood_severity'] > 0.7:
                        st.error(f"🆘 @{tweet['username'][:10]}\n{tweet['text'][:50]}...")
                    elif tweet['flood_severity'] > 0.4:
                        st.warning(f"⚠️ @{tweet['username'][:10]}\n{tweet['text'][:50]}...")
                    elif tweet['flood_severity'] > 0:
                        st.info(f"🌧️ @{tweet['username'][:10]}\n{tweet['text'][:50]}...")
                    else:
                        st.success(f"☀️ @{tweet['username'][:10]}\n{tweet['text'][:50]}...")
                else:
                    st.write(f"📱 @{tweet['username'][:10]}\n{tweet['text'][:50]}...")

        else:
            st.info("No tweets yet!")
    else:
        st.info("Tweet stream disabled.")

# -------------------------------
# Bottom statistics
st.subheader("📊 Comprehensive Simulation Dashboard")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Global Water Level", f"{water_level} m")
with col2:
    st.metric("Total Flooded Area", f"{flooded_area_pct:.1f}%")
with col3:
    st.metric("Elevation Range", f"{np.nanmin(elev_array):.0f}m - {np.nanmax(elev_array):.0f}m")
with col4:
    tweets_count = len(st.session_state.tweet_generator.get_recent_tweets(100))
    st.metric("Total Tweets", tweets_count)
with col5:
    if show_sensors and sensor_summary:
        st.metric("Active Sensors", sensor_summary['operational'])
    else:
        st.metric("Active Sensors", "0")
with col6:
    st.metric("Local Flood Events", len(st.session_state.flood_manager.flood_events))

# Local flood events summary
if st.session_state.flood_manager.flood_events:
    st.subheader("🎯 Local Flood Events Summary")

    flood_data = []
    for event in st.session_state.flood_manager.flood_events:
        flood_data.append({
            'Event ID': event['id'],
            'Location': f"{event['lat']:.4f}, {event['lon']:.4f}",
            'Water Level (m)': f"{event['water_level']:.1f}",
            'Radius (km)': f"{event['radius_km']:.1f}",
            'Created': event['created_at'].strftime('%H:%M:%S'),
            'Status': 'Active' if event['active'] else 'Inactive'
        })

    flood_df = pd.DataFrame(flood_data)
    st.dataframe(flood_df, use_container_width=True)

# Detailed sensor data table
if show_sensors and sensors:
    st.subheader("📋 Detailed Sensor Readings")

    # Create a DataFrame for better display
    sensor_data = []
    for sensor in sensors:
        sensor_data.append({
            'ID': sensor['id'],
            'Type': sensor['type'],
            'Status': sensor['status'],
            'Alert Level': sensor['alert_level'],
            'Current Reading': f"{sensor['current_reading']:.2f}",
            'Water Depth (m)': f"{sensor.get('water_depth', 0):.1f}",
            'Elevation (m)': f"{sensor['elevation']:.1f}",
            'Location': f"{sensor['lat']:.4f}, {sensor['lon']:.4f}"
        })

    # Display as table
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
            extent, combined_flood_mask, elev_array, water_level, city_name
        )
        st.session_state.tweet_generator.add_tweet_to_stream(tweet)

    # Add an auto-refresh toggle in sidebar
    auto_refresh = st.sidebar.checkbox("Auto-refresh tweets", value=True)
    if auto_refresh:
        time.sleep(3)
        st.rerun()