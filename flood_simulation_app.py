# Add this at the very top to suppress urllib3 warnings
import warnings
import urllib3
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

# Optional imports with fallbacks
try:
    import py3dep
    ELEVATION_AVAILABLE = True
except ImportError:
    ELEVATION_AVAILABLE = False
    st.warning("py3dep not available - using synthetic elevation data")

try:
    import osmnx as ox
    GEOCODING_AVAILABLE = True
except ImportError:
    GEOCODING_AVAILABLE = False
    st.warning("osmnx not available - using default coordinates")

try:
    import geopandas as gpd
    from shapely.geometry import box
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False
    st.warning("geopandas/shapely not available - using simple geometry")

warnings.filterwarnings('ignore')
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
                'is_flooded': False,
                'battery_level': random.uniform(80, 100),
                'signal_strength': random.uniform(70, 100)
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
                    'battery_level': str(sensor.get('battery_level', 100)),
                    'signal_strength': str(sensor.get('signal_strength', 100)),
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

def get_location_bounds(place: str) -> Tuple[List[float], str]:
    """Get location bounds with multiple fallback methods"""

    # Predefined locations for common places
    default_locations = {
        "dallas": [32.7157, 32.8074, -96.8716, -96.7297],  # [lat_min, lat_max, lon_min, lon_max]
        "houston": [29.5274, 29.9748, -95.8233, -95.0140],
        "austin": [30.0986, 30.5081, -97.9383, -97.5675],
        "san antonio": [29.2140, 29.6605, -98.7598, -98.2940],
        "new york": [40.4774, 40.9176, -74.2591, -73.7004],
        "los angeles": [33.7037, 34.3373, -118.6681, -118.1553],
        "chicago": [41.6444, 42.0230, -87.9402, -87.5240],
        "miami": [25.7617, 25.8557, -80.3148, -80.1918],
        "seattle": [47.4810, 47.7341, -122.4594, -122.2244],
        "denver": [39.6147, 39.9142, -105.1102, -104.8009]
    }

    place_key = place.lower().split(',')[0].strip()

    if place_key in default_locations:
        bounds = default_locations[place_key]
        return bounds, place_key.title()

    # Try geocoding if available
    if GEOCODING_AVAILABLE:
        try:
            place_gdf = ox.geocode_to_gdf(place)
            place_geom = place_gdf.geometry[0]
            bounds_tuple = place_geom.bounds
            bounds = [bounds_tuple[1], bounds_tuple[3], bounds_tuple[0], bounds_tuple[2]]  # Convert to [lat_min, lat_max, lon_min, lon_max]
            return bounds, place
        except Exception as e:
            st.warning(f"Geocoding failed: {e}")

    # Final fallback - use Dallas
    st.warning(f"Using Dallas coordinates as fallback for '{place}'")
    return default_locations["dallas"], "Dallas (fallback)"

def create_synthetic_elevation(bounds: List[float], size: int = 100) -> Tuple[np.ndarray, List[float]]:
    """Create synthetic elevation data"""
    lat_min, lat_max, lon_min, lon_max = bounds

    # Create coordinate arrays
    lats = np.linspace(lat_min, lat_max, size)
    lons = np.linspace(lon_min, lon_max, size)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Create realistic elevation pattern
    # Base elevation with some randomness
    base_elevation = 100 + np.random.normal(0, 20, (size, size))

    # Add some hills and valleys
    center_lat, center_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
    distance_from_center = np.sqrt((lat_grid - center_lat)**2 + (lon_grid - center_lon)**2)

    # Create elevation variations
    hill_pattern = 50 * np.sin(distance_from_center * 20) * np.exp(-distance_from_center * 10)
    valley_pattern = -30 * np.cos(lat_grid * 50) * np.cos(lon_grid * 50)

    # Combine patterns
    elevation = base_elevation + hill_pattern + valley_pattern

    # Ensure reasonable elevation range (0-300m)
    elevation = np.clip(elevation, 0, 300)

    # Create extent
    extent = [lon_min, lon_max, lat_min, lat_max]

    return elevation, extent

def get_elevation_data(place: str, bounds: List[float]) -> Tuple[np.ndarray, List[float]]:
    """Get elevation data with multiple fallback methods"""

    if ELEVATION_AVAILABLE:
        try:
            st.info("🌍 Attempting to download real elevation data...")

            if GEO_AVAILABLE:
                # Try with geometry
                place_gdf = ox.geocode_to_gdf(place)
                place_geom = place_gdf.geometry[0]
                dem_data = py3dep.get_dem(geometry=place_geom, resolution=30, crs="EPSG:4326")
            else:
                # Try with bounding box
                lat_min, lat_max, lon_min, lon_max = bounds
                bbox = [lat_min, lon_min, lat_max, lon_max]  # py3dep format
                dem_data = py3dep.get_dem(bbox, resolution=90)

            elev_array = dem_data.values.squeeze() if hasattr(dem_data, 'values') else dem_data.squeeze()

            if hasattr(dem_data, 'rio'):
                dem_bounds = dem_data.rio.bounds()
                extent = [dem_bounds[0], dem_bounds[2], dem_bounds[1], dem_bounds[3]]
            else:
                extent = [bounds[2], bounds[3], bounds[0], bounds[1]]  # Convert back to [lon_min, lon_max, lat_min, lat_max]

            # Validate data
            if elev_array is None or elev_array.size == 0:
                raise ValueError("Empty elevation data")

            nan_count = np.isnan(elev_array).sum()
            if nan_count == elev_array.size:
                raise ValueError("All elevation values are NaN")
            elif nan_count > 0:
                elev_array = np.where(np.isnan(elev_array), np.nanmean(elev_array), elev_array)

            st.success("✅ Real elevation data downloaded successfully!")
            return elev_array, extent

        except Exception as e:
            st.warning(f"⚠️ Real elevation data failed: {e}")
            st.info("🔄 Falling back to synthetic elevation data...")

    # Fallback to synthetic data
    st.info("🎨 Creating synthetic elevation data...")
    elevation, extent = create_synthetic_elevation(bounds)
    st.success("✅ Synthetic elevation data created!")
    return elevation, extent

# -------------------------------
# Streamlit Setup
st.set_page_config(layout="wide", page_title="Flood Simulation", page_icon="🌊")
st.title("🌊 Interactive Flood Simulation with Real-Time Data Streaming")

# Initialize components in session state
if 'tweet_generator' not in st.session_state:
    st.session_state.tweet_generator = FloodTweetGenerator()

if 'sensor_network' not in st.session_state:
    st.session_state.sensor_network = FloodSensorNetwork()

if 'flood_manager' not in st.session_state:
    st.session_state.flood_manager = LocalFloodManager()

# -------------------------------
# User Inputs
st.sidebar.header("🌊 Flood Simulation Settings")
place = st.sidebar.text_input("Enter Location", "Dallas, Texas, USA")
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

# Force refresh button in sidebar
if st.sidebar.button("🔄 Force Refresh", type="primary"):
    st.rerun()

# -------------------------------
# Get location bounds and elevation data
try:
    # Get location bounds
    with st.spinner(f"📍 Looking up location: {place}..."):
        bounds, resolved_place = get_location_bounds(place)
        st.success(f"📍 Location resolved: {resolved_place}")

    # Get elevation data with progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("🌍 Fetching elevation data...")
    progress_bar.progress(25)

    elev_array, extent = get_elevation_data(place, bounds)

    progress_bar.progress(75)
    status_text.text("✅ Processing elevation data...")

    # Store in session state
    st.session_state.place = resolved_place
    st.session_state.bounds = bounds
    st.session_state.elev_array = elev_array
    st.session_state.extent = extent
    st.session_state.water_level = water_level

    progress_bar.progress(100)
    status_text.text("✅ Elevation data ready!")

    # Clear progress indicators
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    st.info(f"📊 Elevation range: {np.nanmin(elev_array):.1f}m to {np.nanmax(elev_array):.1f}m")

except Exception as e:
    st.error(f"❌ Critical error loading data: {e}")
    st.info("💡 Try using a simpler location name like 'Dallas' or 'Houston'")
    st.stop()

# -------------------------------
# Flood Simulation
global_flood_mask = elev_array <= water_level
flooded_area_pct = (np.sum(global_flood_mask) / global_flood_mask.size) * 100

# Store flood mask in session state
st.session_state.flood_mask = global_flood_mask

st.success(f"💧 Simulation ready! Flooded area: {flooded_area_pct:.1f}% of the region")

# -------------------------------
# Deploy and update sensors
if show_sensors:
    # Deploy sensors if not already deployed or if number changed
    if not hasattr(st.session_state.sensor_network, 'sensors') or len(st.session_state.sensor_network.sensors) != num_sensors:
        with st.spinner("📡 Deploying sensor network..."):
            st.session_state.sensor_network.num_sensors = num_sensors
            sensors = st.session_state.sensor_network.deploy_sensors(extent, elev_array)
            st.success(f"📡 Deployed {len(sensors)} water level sensors")

    # Update sensor readings based on current flood conditions
    sensors = st.session_state.sensor_network.update_sensor_readings(extent, global_flood_mask, elev_array, water_level)
    sensor_summary = st.session_state.sensor_network.get_sensor_summary()

    # Stream sensor data to Redis
    if st.session_state.tweet_generator.redis_connected:
        st.session_state.sensor_network.add_sensor_data_to_stream(
            st.session_state.tweet_generator.redis_client,
            "sensor_data"
        )
else:
    sensors = []
    sensor_summary = {}

# -------------------------------
# Create layout
col1, col2 = st.columns([4, 1])

with col1:
    st.subheader("🗺️ Flood Simulation Map with Sensor Network")

    if use_interactive_map:
        # Create interactive Plotly map
        try:
            fig = go.Figure()

            # Custom terrain colorscale
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
                            name=f'{alert_level.title()} Sensors ({len(level_sensors)})',
                            text=[f"{s['id']}<br>Reading: {s['current_reading']:.2f}m<br>Status: {s['status']}<br>Water Depth: {s.get('water_depth', 0):.2f}m"
                                  for s in level_sensors],
                            hovertemplate='%{text}<extra></extra>'
                        ))

            # Configure layout
            fig.update_layout(
                title=f"Flood Map: {resolved_place}<br>Water Level: {water_level}m | Flooded Area: {flooded_area_pct:.1f}%",
                xaxis_title="Longitude (°)",
                yaxis_title="Latitude (°)",
                height=700,
                showlegend=True,
                legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
                hovermode='closest'
            )

            # Equal aspect ratio
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

            # Display the interactive map
            st.plotly_chart(fig, use_container_width=True, key="flood_map")

        except Exception as e:
            st.error(f"Interactive map error: {e}")
            use_interactive_map = False

    # Fallback static map
    if not use_interactive_map:
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
                sensor_groups = {'normal': [], 'caution': [], 'warning': [], 'critical': []}
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
            ax.set_title(f"Flood Simulation: {resolved_place}\nWater Level: {water_level}m | Flooded Area: {flooded_area_pct:.1f}%")
            ax.grid(True, alpha=0.3)

            if show_sensors and sensors:
                ax.legend(loc='upper left', title='Sensor Status', frameon=True, fancybox=True, shadow=True)

            st.pyplot(fig, clear_figure=True, use_container_width=True)
            plt.close(fig)

        except Exception as e:
            st.error(f"Map rendering error: {e}")

with col2:
    st.subheader("📊 Live Status")

    # Redis connection status
    if st.session_state.tweet_generator.redis_connected:
        st.success("🟢 Redis Connected")
    else:
        st.error("🔴 Redis Disconnected")
        st.caption("Install Redis: `pip install redis && redis-server`")

    st.write("---")

    # Sensor statistics
    if show_sensors and sensor_summary:
        st.subheader("📡 Sensor Network")
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.metric("Total", sensor_summary['total_sensors'])
            st.metric("Critical", sensor_summary['critical_alerts'])

        with col_s2:
            st.metric("Operational", sensor_summary['operational'])
            st.metric("Warning", sensor_summary['warning_alerts'])

    st.write("---")

    # Tweet stream statistics
    if tweet_enabled:
        st.subheader("🐦 Tweet Stream")
        tweet_stats = st.session_state.tweet_generator.get_stream_stats()

        if tweet_stats['connected']:
            st.metric("Total Tweets", tweet_stats['total_messages'])
            st.metric("Rate/min", tweet_rate)

            if flooded_area_pct > 1:
                st.caption("🌊 Flood mode active")
            else:
                st.caption("☀️ Normal conditions")
        else:
            st.error("Stream unavailable")

    # Quick stats
    st.write("---")
    st.subheader("📈 Quick Stats")
    st.write(f"🌍 **Elevation**: {np.nanmin(elev_array):.0f}m - {np.nanmax(elev_array):.0f}m")
    st.write(f"💧 **Water Level**: {water_level}m")
    st.write(f"🌊 **Flooded**: {flooded_area_pct:.1f}%")

# -------------------------------
# Background data generation
if st.session_state.tweet_generator.redis_connected:
    city_name = resolved_place.split(',')[0]

    # Generate continuous tweets
    if tweet_enabled:
        new_tweets = st.session_state.tweet_generator.generate_continuous_tweets(
            extent, global_flood_mask, elev_array, water_level, city_name, tweet_rate
        )

        # Stream tweets to Redis
        for tweet in new_tweets:
            st.session_state.tweet_generator.add_tweet_to_stream(tweet)

# -------------------------------
# Bottom dashboard
st.subheader("📊 Simulation Dashboard")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Water Level", f"{water_level}m")
with col2:
    st.metric("Flooded Area", f"{flooded_area_pct:.1f}%")
with col3:
    st.metric("Elevation Range", f"{np.nanmax(elev_array) - np.nanmin(elev_array):.0f}m")
with col4:
    if tweet_enabled and st.session_state.tweet_generator.redis_connected:
        tweet_stats = st.session_state.tweet_generator.get_stream_stats()
        st.metric("Tweets", tweet_stats['total_messages'])
    else:
        st.metric("Tweets", "Disabled")
with col5:
    if show_sensors and sensor_summary:
        st.metric("Active Sensors", f"{sensor_summary['operational']}/{sensor_summary['total_sensors']}")
    else:
        st.metric("Sensors", "Disabled")
with col6:
    critical_count = sensor_summary.get('critical_alerts', 0) if sensor_summary else 0
    st.metric("Critical Alerts", critical_count, delta=critical_count if critical_count > 0 else None)

# Sensor details table
if show_sensors and sensors:
    with st.expander("📋 Detailed Sensor Data", expanded=False):
        sensor_data = []
        for sensor in sensors:
            sensor_data.append({
                'Sensor ID': sensor['id'],
                'Location': f"{sensor['lat']:.4f}, {sensor['lon']:.4f}",
                'Elevation': f"{sensor['elevation']:.1f}m",
                'Reading': f"{sensor['current_reading']:.2f}m",
                'Water Depth': f"{sensor.get('water_depth', 0):.2f}m",
                'Alert Level': sensor['alert_level'],
                'Status': sensor['status'],
                'Battery': f"{sensor.get('battery_level', 100):.0f}%",
                'Signal': f"{sensor.get('signal_strength', 100):.0f}%"
            })

        df = pd.DataFrame(sensor_data)
        st.dataframe(df, use_container_width=True)

# Information panels
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.info("📡 **Data Streaming**: Sensors update every 10 seconds, tweets generate based on flood conditions")

with col_info2:
    st.info("🐦 **Tweet Generation**: Realistic social media simulation with flood-aware content generation")

# Warning and auto-refresh
st.warning("⚠️ Educational simulation only. Real flood modeling requires meteorological data, drainage systems, and temporal dynamics.")

# Auto-refresh for continuous operation
if tweet_enabled or show_sensors:
    time.sleep(2)  # Update every 2 seconds for responsive streaming
    st.rerun()