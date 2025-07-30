import streamlit as st
import redis
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.interpolate import griddata
import warnings
import uuid
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import logging

warnings.filterwarnings('ignore')

# Import the multi-agent components
from enum import Enum

class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AgentType(Enum):
    SENSOR_AGENT = "sensor_agent"
    TWEET_AGENT = "tweet_agent"
    COORDINATION_AGENT = "coordination_agent"
    COMMUNICATION_AGENT = "communication_agent"

@dataclass
class FloodAlert:
    id: str
    source: str  # 'sensor' or 'tweet'
    location: dict  # {'lat': x, 'lon': y}
    severity: float  # 0.0 to 1.0
    alert_level: AlertLevel
    timestamp: str
    details: dict
    confidence: float
    area_radius: float = 0.5  # km

class BaseAgent:
    def __init__(self, agent_id: str, agent_type: AgentType, redis_host='localhost', redis_port=6379):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.running = False
        self.message_handlers = {}

        # Agent communication channels
        self.inbox_channel = f"agent:{agent_id}:inbox"
        self.broadcast_channel = "agents:broadcast"

        # Initialize pubsub for agent communication
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe(self.inbox_channel, self.broadcast_channel)

    def send_message(self, recipient_id: str, message_type: str, data: dict):
        """Send message to another agent"""
        message = {
            'from': self.agent_id,
            'type': message_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }

        if recipient_id == "broadcast":
            self.redis_client.publish(self.broadcast_channel, json.dumps(message))
        else:
            self.redis_client.publish(f"agent:{recipient_id}:inbox", json.dumps(message))

class SensorFloodAnalyzer(BaseAgent):
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        """Initialize the sensor-based flood analyzer with multi-agent capabilities"""
        # Initialize base agent
        super().__init__("sensor_analysis_agent", AgentType.SENSOR_AGENT, redis_host, redis_port)

        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True
            )
            self.redis_client.ping()
            self.connected = True
        except Exception as e:
            st.error(f"❌ Redis connection failed: {e}")
            self.connected = False

        self.sensor_stream = "sensor_data"
        self.sensor_buffer = deque(maxlen=2000)  # Keep last 2000 sensor readings

        # Track sent alerts to avoid duplicates
        self.sent_alerts = set()
        self.alert_cooldown = {}  # Track cooldowns for areas
        self.last_analysis_time = datetime.now()

        # Sensor analysis thresholds
        self.water_depth_thresholds = {
            'normal': 0.3,      # Below 30cm
            'caution': 0.6,     # 30-60cm
            'warning': 1.0,     # 60cm-1m
            'critical': 1.5,    # Above 1.5m
            'emergency': 2.0    # Above 2m
        }

        self.reading_thresholds = {
            'normal': 2.0,      # Normal sensor reading
            'elevated': 3.5,    # Elevated reading
            'high': 5.0,        # High reading
            'critical': 7.0,    # Critical reading
            'emergency': 10.0   # Emergency reading
        }

        # Analysis parameters
        self.network_analysis_params = {
            'cluster_eps': 0.01,        # Clustering distance threshold
            'min_cluster_size': 2,      # Minimum sensors per cluster
            'interpolation_grid': 50,   # Grid size for interpolation
            'trend_window_hours': 6,    # Hours for trend analysis
            'prediction_horizon': 2     # Hours to predict ahead
        }

        # Alert generation settings
        self.alert_settings = {
            'min_severity_threshold': 0.4,  # Minimum severity to send alert
            'cooldown_minutes': 15,          # Minutes between alerts for same area
            'zone_expansion_factor': 1.2     # Factor for zone expansion detection
        }

        print(f"Sensor Analysis Agent initialized with ID: {self.agent_id}")

    def connect_to_redis(self):
        """Attempt to reconnect to Redis"""
        try:
            self.redis_client = redis.Redis(decode_responses=True)
            self.redis_client.ping()
            self.connected = True
            return True
        except:
            self.connected = False
            return False

    def classify_sensor_status(self, sensor_data):
        """Classify sensor status based on readings and water depth"""
        current_reading = sensor_data.get('current_reading', 0)
        water_depth = sensor_data.get('water_depth', 0)

        # Determine status based on water depth (primary indicator)
        if water_depth >= self.water_depth_thresholds['emergency']:
            status = 'emergency'
            severity = 1.0
        elif water_depth >= self.water_depth_thresholds['critical']:
            status = 'critical'
            severity = 0.8
        elif water_depth >= self.water_depth_thresholds['warning']:
            status = 'warning'
            severity = 0.6
        elif water_depth >= self.water_depth_thresholds['caution']:
            status = 'caution'
            severity = 0.4
        else:
            status = 'normal'
            severity = 0.1

        # Adjust based on sensor reading (secondary indicator)
        if current_reading >= self.reading_thresholds['emergency']:
            severity = min(severity + 0.2, 1.0)
        elif current_reading >= self.reading_thresholds['critical']:
            severity = min(severity + 0.15, 1.0)
        elif current_reading >= self.reading_thresholds['high']:
            severity = min(severity + 0.1, 1.0)

        return status, severity

    def fetch_sensor_data_with_timeframe(self, count=500, hours_back=24):
        """Fetch sensor data from Redis stream within specified timeframe"""
        if not self.connected:
            return []

        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            start_timestamp = int(start_time.timestamp() * 1000)

            messages = self.redis_client.xrevrange(self.sensor_stream, count=count)
            sensors = []

            for message_id, fields in messages:
                # Parse timestamp from message ID
                try:
                    msg_timestamp = int(message_id.split('-')[0])
                    if msg_timestamp < start_timestamp:
                        continue
                except:
                    pass

                sensor_data = {
                    'id': message_id,
                    'sensor_id': fields.get('sensor_id', ''),
                    'lat': float(fields.get('lat', 0)),
                    'lon': float(fields.get('lon', 0)),
                    'current_reading': float(fields.get('current_reading', 0)),
                    'water_depth': float(fields.get('water_depth', 0)),
                    'alert_level': fields.get('alert_level', 'normal'),
                    'is_flooded': fields.get('is_flooded', 'False') == 'True',
                    'timestamp': fields.get('timestamp', ''),
                    'battery_level': float(fields.get('battery_level', 100)),
                    'signal_strength': float(fields.get('signal_strength', 100))
                }

                # Add calculated status and severity
                status, severity = self.classify_sensor_status(sensor_data)
                sensor_data['calculated_status'] = status
                sensor_data['severity_score'] = severity

                sensors.append(sensor_data)

            return sensors
        except Exception as e:
            st.error(f"Error fetching sensor data: {e}")
            return []

    def detect_flood_zones(self, sensors, eps=0.01, min_samples=2):
        """Detect flood zones using sensor clustering and interpolation"""
        if len(sensors) < min_samples:
            return []

        # Filter for sensors with concerning readings
        flood_sensors = [s for s in sensors if
                         s['severity_score'] > 0.3 or
                         s['water_depth'] > self.water_depth_thresholds['caution']]

        if len(flood_sensors) < min_samples:
            return []

        # Prepare data for clustering
        coords = np.array([[s['lat'], s['lon']] for s in flood_sensors])
        severities = np.array([s['severity_score'] for s in flood_sensors])
        water_depths = np.array([s['water_depth'] for s in flood_sensors])

        # Weight coordinates by severity and water depth
        weights = (severities + water_depths / 3.0)  # Normalize water depth
        weighted_coords = coords * (1 + weights.reshape(-1, 1))

        # Standardize coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(weighted_coords)

        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(coords_scaled)

        # Analyze clusters
        zones = []
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise points
                continue

            cluster_sensors = [flood_sensors[i] for i in range(len(flood_sensors))
                               if cluster_labels[i] == cluster_id]

            if len(cluster_sensors) >= min_samples:
                # Calculate zone statistics
                lats = [s['lat'] for s in cluster_sensors]
                lons = [s['lon'] for s in cluster_sensors]
                severities = [s['severity_score'] for s in cluster_sensors]
                depths = [s['water_depth'] for s in cluster_sensors]
                readings = [s['current_reading'] for s in cluster_sensors]

                # Calculate zone boundaries
                lat_min, lat_max = min(lats), max(lats)
                lon_min, lon_max = min(lons), max(lons)

                zone_info = {
                    'id': f'zone_{cluster_id}',
                    'center_lat': np.mean(lats),
                    'center_lon': np.mean(lons),
                    'bounds': {
                        'lat_min': lat_min, 'lat_max': lat_max,
                        'lon_min': lon_min, 'lon_max': lon_max
                    },
                    'area_km2': self.calculate_zone_area(lat_min, lat_max, lon_min, lon_max),
                    'sensor_count': len(cluster_sensors),
                    'avg_severity': np.mean(severities),
                    'max_severity': np.max(severities),
                    'avg_water_depth': np.mean(depths),
                    'max_water_depth': np.max(depths),
                    'avg_reading': np.mean(readings),
                    'max_reading': np.max(readings),
                    'flooded_sensors': len([s for s in cluster_sensors if s['is_flooded']]),
                    'critical_sensors': len([s for s in cluster_sensors if s['calculated_status'] in ['critical', 'emergency']]),
                    'zone_status': self.determine_zone_status(cluster_sensors),
                    'confidence': min(len(cluster_sensors) / 5.0, 1.0),
                    'trend': self.calculate_zone_trend(cluster_sensors),
                    'prediction': self.predict_zone_development(cluster_sensors),
                    'sensors': cluster_sensors
                }
                zones.append(zone_info)

        # Sort by severity and confidence
        zones.sort(key=lambda x: (x['max_severity'] * x['confidence']), reverse=True)

        # MULTI-AGENT INTEGRATION: Send alerts for significant zones
        self.send_zone_alerts(zones)

        return zones

    def send_zone_alerts(self, zones):
        """Send flood alerts to coordination agent for significant zones"""
        current_time = datetime.now()

        for zone in zones:
            # Check if we should send an alert for this zone
            if self.should_send_zone_alert(zone, current_time):
                # Create flood alert
                alert = FloodAlert(
                    id=str(uuid.uuid4()),
                    source='sensor',
                    location={
                        'lat': zone['center_lat'],
                        'lon': zone['center_lon']
                    },
                    severity=zone['max_severity'],
                    alert_level=self.map_zone_status_to_alert_level(zone['zone_status']),
                    timestamp=current_time.isoformat(),
                    details={
                        'zone_id': zone['id'],
                        'sensor_count': zone['sensor_count'],
                        'area_km2': zone['area_km2'],
                        'max_water_depth': zone['max_water_depth'],
                        'critical_sensors': zone['critical_sensors'],
                        'trend': zone['trend'],
                        'prediction': zone['prediction'],
                        'zone_status': zone['zone_status']
                    },
                    confidence=zone['confidence'],
                    area_radius=np.sqrt(zone['area_km2'] / np.pi)  # Convert area to radius
                )

                # Send to coordination agent
                try:
                    self.send_message("coordination_agent", "flood_alert", asdict(alert))

                    # Track sent alert and set cooldown
                    zone_key = f"{zone['center_lat']:.3f},{zone['center_lon']:.3f}"
                    self.sent_alerts.add(alert.id)
                    self.alert_cooldown[zone_key] = current_time

                    print(f"🚨 Sent flood alert for zone {zone['id']} to coordination agent")
                    print(f"   Severity: {zone['max_severity']:.2f}, Status: {zone['zone_status']}")

                except Exception as e:
                    print(f"Error sending alert to coordination agent: {e}")

    def should_send_zone_alert(self, zone, current_time):
        """Determine if we should send an alert for this zone"""
        # Check minimum severity threshold
        if zone['max_severity'] < self.alert_settings['min_severity_threshold']:
            return False

        # Check cooldown for this area
        zone_key = f"{zone['center_lat']:.3f},{zone['center_lon']:.3f}"
        if zone_key in self.alert_cooldown:
            last_alert_time = self.alert_cooldown[zone_key]
            cooldown_delta = timedelta(minutes=self.alert_settings['cooldown_minutes'])
            if current_time - last_alert_time < cooldown_delta:
                return False

        # Send alert for critical/emergency zones always
        if zone['zone_status'] in ['critical', 'emergency']:
            return True

        # Send alert for zones showing significant worsening trend
        if zone['trend'] > 0.3 and zone['max_severity'] > 0.5:
            return True

        # Send alert for new zones (not in cooldown)
        if zone_key not in self.alert_cooldown:
            return True

        return False

    def map_zone_status_to_alert_level(self, zone_status):
        """Map zone status to AlertLevel enum"""
        mapping = {
            'normal': AlertLevel.LOW,
            'caution': AlertLevel.LOW,
            'warning': AlertLevel.MEDIUM,
            'critical': AlertLevel.HIGH,
            'emergency': AlertLevel.CRITICAL
        }
        return mapping.get(zone_status, AlertLevel.LOW)

    def send_individual_sensor_alerts(self, sensors):
        """Send alerts for individual critical sensors not part of zones"""
        current_time = datetime.now()

        critical_sensors = [s for s in sensors if
                            s['calculated_status'] in ['critical', 'emergency'] and
                            s['severity_score'] >= 0.7]

        for sensor in critical_sensors:
            sensor_key = f"sensor_{sensor['sensor_id']}"

            # Check cooldown
            if sensor_key in self.alert_cooldown:
                last_alert_time = self.alert_cooldown[sensor_key]
                cooldown_delta = timedelta(minutes=self.alert_settings['cooldown_minutes'])
                if current_time - last_alert_time < cooldown_delta:
                    continue

            # Create individual sensor alert
            alert = FloodAlert(
                id=str(uuid.uuid4()),
                source='sensor',
                location={
                    'lat': sensor['lat'],
                    'lon': sensor['lon']
                },
                severity=sensor['severity_score'],
                alert_level=self.map_sensor_status_to_alert_level(sensor['calculated_status']),
                timestamp=current_time.isoformat(),
                details={
                    'sensor_id': sensor['sensor_id'],
                    'water_depth': sensor['water_depth'],
                    'current_reading': sensor['current_reading'],
                    'battery_level': sensor['battery_level'],
                    'signal_strength': sensor['signal_strength'],
                    'is_flooded': sensor['is_flooded']
                },
                confidence=0.9,  # High confidence for direct sensor readings
                area_radius=0.2  # Small radius for individual sensors
            )

            try:
                self.send_message("coordination_agent", "flood_alert", asdict(alert))
                self.alert_cooldown[sensor_key] = current_time

                print(f"🚨 Sent critical sensor alert for {sensor['sensor_id']}")
                print(f"   Status: {sensor['calculated_status']}, Depth: {sensor['water_depth']:.2f}m")

            except Exception as e:
                print(f"Error sending sensor alert: {e}")

    def map_sensor_status_to_alert_level(self, sensor_status):
        """Map sensor status to AlertLevel enum"""
        mapping = {
            'normal': AlertLevel.LOW,
            'caution': AlertLevel.LOW,
            'warning': AlertLevel.MEDIUM,
            'critical': AlertLevel.HIGH,
            'emergency': AlertLevel.CRITICAL
        }
        return mapping.get(sensor_status, AlertLevel.LOW)

    def send_network_status_update(self, analysis):
        """Send network status update to other agents"""
        try:
            status_update = {
                'agent_id': self.agent_id,
                'timestamp': datetime.now().isoformat(),
                'network_stats': analysis['network_stats'],
                'network_risk': analysis['network_risk'],
                'active_zones': len(analysis['flood_zones']),
                'predictions': analysis['predictions']
            }

            self.send_message("broadcast", "sensor_network_update", status_update)
            print(f"📊 Sent network status update to all agents")

        except Exception as e:
            print(f"Error sending network status update: {e}")

    # Include all the existing methods from the original class
    def calculate_zone_area(self, lat_min, lat_max, lon_min, lon_max):
        """Calculate approximate area of zone in km²"""
        # Rough approximation using Haversine-like calculation
        lat_km = (lat_max - lat_min) * 111  # 1 degree lat ≈ 111 km
        lon_km = (lon_max - lon_min) * 111 * np.cos(np.radians((lat_min + lat_max) / 2))
        return lat_km * lon_km

    def determine_zone_status(self, sensors):
        """Determine overall status of a flood zone"""
        emergency_count = len([s for s in sensors if s['calculated_status'] == 'emergency'])
        critical_count = len([s for s in sensors if s['calculated_status'] == 'critical'])
        warning_count = len([s for s in sensors if s['calculated_status'] == 'warning'])

        total_sensors = len(sensors)

        if emergency_count / total_sensors > 0.5:
            return 'emergency'
        elif (emergency_count + critical_count) / total_sensors > 0.4:
            return 'critical'
        elif (warning_count + critical_count + emergency_count) / total_sensors > 0.3:
            return 'warning'
        else:
            return 'caution'

    def calculate_zone_trend(self, sensors):
        """Calculate trend for zone (positive = worsening, negative = improving)"""
        if len(sensors) < 3:
            return 0.0

        try:
            # Group sensors by time and calculate average severity
            time_severity = []
            for sensor in sensors:
                if sensor['timestamp']:
                    try:
                        ts = datetime.fromisoformat(sensor['timestamp'].replace('Z', '+00:00'))
                        time_severity.append((ts.timestamp(), sensor['severity_score']))
                    except:
                        continue

            if len(time_severity) < 3:
                return 0.0

            time_severity.sort()  # Sort by time
            times = np.array([t[0] for t in time_severity])
            severities = np.array([t[1] for t in time_severity])

            # Calculate linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(times, severities)

            # Normalize slope to -1 to 1 range
            normalized_slope = np.tanh(slope * 3600)  # Scale by hour
            return normalized_slope

        except:
            return 0.0

    def predict_zone_development(self, sensors):
        """Predict how zone will develop in next few hours"""
        if not sensors:
            return {'next_hour_severity': 0.0, 'peak_expected': False, 'stabilizing': False}

        current_avg_severity = np.mean([s['severity_score'] for s in sensors])
        trend = self.calculate_zone_trend(sensors)

        # Simple linear extrapolation for next hour
        next_hour_severity = max(0.0, min(1.0, current_avg_severity + trend * 0.1))

        # Determine if peak is expected (high severity with positive trend)
        peak_expected = current_avg_severity > 0.7 and trend > 0.2

        # Determine if stabilizing (trend near zero or negative with moderate severity)
        stabilizing = abs(trend) < 0.1 or (trend < 0 and current_avg_severity < 0.6)

        return {
            'next_hour_severity': next_hour_severity,
            'peak_expected': peak_expected,
            'stabilizing': stabilizing,
            'trend_strength': abs(trend)
        }

    def analyze_sensor_network(self, sensors):
        """Comprehensive sensor network analysis with multi-agent integration"""
        if not sensors:
            return self._empty_analysis()

        # Basic statistics
        total_sensors = len(sensors)
        active_sensors = len([s for s in sensors if s['alert_level'] != 'offline'])
        flooded_sensors = len([s for s in sensors if s['is_flooded']])

        # Status distribution
        status_counts = {}
        for status in ['normal', 'caution', 'warning', 'critical', 'emergency']:
            status_counts[status] = len([s for s in sensors if s['calculated_status'] == status])

        # Severity analysis
        severities = [s['severity_score'] for s in sensors]
        water_depths = [s['water_depth'] for s in sensors]
        readings = [s['current_reading'] for s in sensors]

        # Network health
        avg_battery = np.mean([s['battery_level'] for s in sensors])
        low_battery_sensors = len([s for s in sensors if s['battery_level'] < 20])
        poor_signal_sensors = len([s for s in sensors if s['signal_strength'] < 30])

        # Detect flood zones (this will also send alerts)
        flood_zones = self.detect_flood_zones(sensors)

        # Send individual sensor alerts for critical sensors not in zones
        self.send_individual_sensor_alerts(sensors)

        # Calculate overall network risk
        network_risk = self.calculate_network_risk(sensors, flood_zones)

        # Generate predictions
        predictions = self.generate_network_predictions(sensors, flood_zones)

        analysis = {
            'timestamp': datetime.now(),
            'network_stats': {
                'total_sensors': total_sensors,
                'active_sensors': active_sensors,
                'flooded_sensors': flooded_sensors,
                'status_distribution': status_counts,
                'avg_severity': np.mean(severities) if severities else 0.0,
                'max_severity': np.max(severities) if severities else 0.0,
                'avg_water_depth': np.mean(water_depths) if water_depths else 0.0,
                'max_water_depth': np.max(water_depths) if water_depths else 0.0,
                'avg_reading': np.mean(readings) if readings else 0.0,
                'max_reading': np.max(readings) if readings else 0.0
            },
            'network_health': {
                'avg_battery_level': avg_battery,
                'low_battery_sensors': low_battery_sensors,
                'poor_signal_sensors': poor_signal_sensors,
                'network_coverage': self.calculate_network_coverage(sensors)
            },
            'flood_zones': flood_zones,
            'network_risk': network_risk,
            'predictions': predictions,
            'recommendations': self.generate_sensor_recommendations(network_risk, flood_zones, sensors),
            'alerts_sent': {
                'zones': len([z for z in flood_zones if z['max_severity'] >= self.alert_settings['min_severity_threshold']]),
                'critical_sensors': len([s for s in sensors if s['calculated_status'] in ['critical', 'emergency']]),
                'total_alerts': len(self.sent_alerts)
            }
        }

        # Send network status update to other agents
        self.send_network_status_update(analysis)

        return analysis

    def _empty_analysis(self):
        """Return empty analysis structure"""
        return {
            'timestamp': datetime.now(),
            'network_stats': {
                'total_sensors': 0,
                'active_sensors': 0,
                'flooded_sensors': 0,
                'status_distribution': {status: 0 for status in ['normal', 'caution', 'warning', 'critical', 'emergency']},
                'avg_severity': 0.0,
                'max_severity': 0.0,
                'avg_water_depth': 0.0,
                'max_water_depth': 0.0,
                'avg_reading': 0.0,
                'max_reading': 0.0
            },
            'network_health': {
                'avg_battery_level': 0.0,
                'low_battery_sensors': 0,
                'poor_signal_sensors': 0,
                'network_coverage': 0.0
            },
            'flood_zones': [],
            'network_risk': {'level': 'low', 'score': 0.0, 'confidence': 0.0},
            'predictions': {},
            'recommendations': [],
            'alerts_sent': {'zones': 0, 'critical_sensors': 0, 'total_alerts': 0}
        }

    def calculate_network_risk(self, sensors, flood_zones):
        """Calculate overall network flood risk assessment"""
        if not sensors:
            return {'level': 'low', 'score': 0.0, 'confidence': 0.0}

        risk_factors = []

        # Factor 1: Percentage of sensors in critical/emergency state
        critical_emergency = len([s for s in sensors if s['calculated_status'] in ['critical', 'emergency']])
        if sensors:
            critical_ratio = critical_emergency / len(sensors)
            risk_factors.append(critical_ratio * 2.0)  # Weight heavily

        # Factor 2: Maximum severity in network
        severities = [s['severity_score'] for s in sensors]
        if severities:
            max_severity = np.max(severities)
            risk_factors.append(max_severity)

        # Factor 3: Flood zone analysis
        if flood_zones:
            zone_risk = np.mean([z['max_severity'] for z in flood_zones])
            zone_count_factor = min(len(flood_zones) / 3.0, 1.0)
            risk_factors.extend([zone_risk, zone_count_factor])

        # Factor 4: Water depth analysis
        water_depths = [s['water_depth'] for s in sensors]
        if water_depths:
            max_depth = np.max(water_depths)
            depth_risk = min(max_depth / 2.0, 1.0)  # Normalize by 2m max
            risk_factors.append(depth_risk)

        # Factor 5: Flooded sensor ratio
        flooded_count = len([s for s in sensors if s['is_flooded']])
        if sensors:
            flooded_ratio = flooded_count / len(sensors)
            risk_factors.append(flooded_ratio)

        # Calculate overall risk
        overall_risk = np.mean(risk_factors) if risk_factors else 0.0

        # Determine risk level
        if overall_risk > 0.8:
            risk_level = 'critical'
        elif overall_risk > 0.6:
            risk_level = 'high'
        elif overall_risk > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        # Calculate confidence based on sensor coverage and health
        active_ratio = len([s for s in sensors if s['alert_level'] != 'offline']) / max(len(sensors), 1)
        network_health = 1.0 - (len([s for s in sensors if s['battery_level'] < 20]) / max(len(sensors), 1))
        confidence = (active_ratio + network_health) / 2.0

        return {
            'level': risk_level,
            'score': overall_risk,
            'confidence': confidence,
            'factors': {
                'critical_sensor_ratio': critical_ratio if 'critical_ratio' in locals() else 0.0,
                'max_severity': max_severity if 'max_severity' in locals() else 0.0,
                'flood_zones': len(flood_zones),
                'max_water_depth': max_depth if 'max_depth' in locals() else 0.0,
                'flooded_sensors': flooded_ratio if 'flooded_ratio' in locals() else 0.0
            }
        }

    def calculate_network_coverage(self, sensors):
        """Calculate network coverage quality"""
        if not sensors:
            return 0.0

        # Simple coverage based on sensor density and signal strength
        active_sensors = [s for s in sensors if s['alert_level'] != 'offline']
        if not active_sensors:
            return 0.0

        avg_signal = np.mean([s['signal_strength'] for s in active_sensors])
        coverage_ratio = len(active_sensors) / len(sensors)

        return (avg_signal / 100.0 + coverage_ratio) / 2.0

    def generate_network_predictions(self, sensors, flood_zones):
        """Generate predictions for network-wide flood development"""
        predictions = {
            'next_hour_risk': 0.0,
            'peak_zones': 0,
            'expanding_zones': 0,
            'new_zones_likely': False,
            'network_degradation_risk': 0.0
        }

        if not sensors:
            return predictions

        # Next hour risk based on trends
        current_severities = [s['severity_score'] for s in sensors]
        if current_severities:
            current_avg = np.mean(current_severities)

            # Calculate overall trend from zones
            if flood_zones:
                trends = [z['trend'] for z in flood_zones]
                avg_trend = np.mean(trends)
                predictions['next_hour_risk'] = max(0.0, min(1.0, current_avg + avg_trend * 0.1))
            else:
                predictions['next_hour_risk'] = current_avg

        # Zone analysis
        if flood_zones:
            predictions['peak_zones'] = len([z for z in flood_zones if z['prediction']['peak_expected']])
            predictions['expanding_zones'] = len([z for z in flood_zones if z['trend'] > 0.2])

            # New zones likely if many sensors showing increasing readings outside current zones
            zone_sensor_ids = set()
            for zone in flood_zones:
                zone_sensor_ids.update([s['sensor_id'] for s in zone['sensors']])

            outside_sensors = [s for s in sensors if s['sensor_id'] not in zone_sensor_ids]
            concerning_outside = len([s for s in outside_sensors if s['severity_score'] > 0.4])
            predictions['new_zones_likely'] = concerning_outside > 2

        # Network degradation risk
        low_battery = len([s for s in sensors if s['battery_level'] < 30])
        poor_signal = len([s for s in sensors if s['signal_strength'] < 40])
        predictions['network_degradation_risk'] = min((low_battery + poor_signal) / len(sensors), 1.0)

        return predictions

    def generate_sensor_recommendations(self, network_risk, flood_zones, sensors):
        """Generate actionable recommendations based on sensor analysis"""
        recommendations = []

        # Risk-based recommendations
        if network_risk['level'] == 'critical':
            recommendations.extend([
                "🚨 CRITICAL FLOOD CONDITIONS DETECTED",
                "🚁 Deploy emergency response teams immediately",
                "📡 Verify critical sensor readings manually",
                "🚧 Initiate evacuation procedures for affected zones",
                "📞 Alert emergency services and authorities"
            ])
        elif network_risk['level'] == 'high':
            recommendations.extend([
                "⚠️ HIGH FLOOD RISK - Prepare emergency response",
                "👥 Stage emergency personnel near affected areas",
                "📊 Increase sensor monitoring frequency",
                "🚨 Issue flood warnings for identified zones",
                "🔧 Check sensor functionality in critical areas"
            ])
        elif network_risk['level'] == 'medium':
            recommendations.extend([
                "🟡 MODERATE FLOOD ACTIVITY - Enhanced monitoring",
                "👀 Monitor identified flood zones closely",
                "🔋 Check battery levels of sensors in flood zones",
                "📈 Analyze trend data for escalation signs",
                "🚰 Inspect drainage systems in affected areas"
            ])
        else:
            recommendations.extend([
                "🟢 NORMAL CONDITIONS - Routine monitoring",
                "🔄 Continue regular sensor maintenance",
                "📊 Monitor for emerging patterns",
                "🔋 Schedule battery replacements as needed"
            ])

        # Zone-specific recommendations
        if flood_zones:
            critical_zones = [z for z in flood_zones if z['zone_status'] in ['critical', 'emergency']]
            if critical_zones:
                recommendations.append(f"🎯 {len(critical_zones)} critical flood zones identified - prioritize these areas")

            expanding_zones = [z for z in flood_zones if z['trend'] > 0.3]
            if expanding_zones:
                recommendations.append(f"📈 {len(expanding_zones)} zones showing expansion - expect escalation")

            large_zones = [z for z in flood_zones if z['area_km2'] > 1.0]
            if large_zones:
                recommendations.append(f"🗺️ {len(large_zones)} large flood zones detected - consider area-wide response")

        # Network health recommendations
        low_battery_count = len([s for s in sensors if s['battery_level'] < 20])
        if low_battery_count > 0:
            recommendations.append(f"🔋 {low_battery_count} sensors with low battery - replace immediately")

        poor_signal_count = len([s for s in sensors if s['signal_strength'] < 30])
        if poor_signal_count > 0:
            recommendations.append(f"📡 {poor_signal_count} sensors with poor signal - check connectivity")

        offline_sensors = len([s for s in sensors if s['alert_level'] == 'offline'])
        if offline_sensors > 0:
            recommendations.append(f"⚠️ {offline_sensors} sensors offline - investigate immediately")

        # Multi-agent coordination recommendations
        if len(self.sent_alerts) > 5:
            recommendations.append(f"📤 {len(self.sent_alerts)} alerts sent to coordination agent - expect response coordination")

        return recommendations

def create_sensor_network_map(sensors, flood_zones, center_lat=32.7767, center_lon=-96.7970):
    """Create interactive sensor network and flood zone map"""
    fig = go.Figure()

    # Set map layout
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11
        ),
        height=600,
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    # Add sensor markers by status
    if sensors:
        status_colors = {
            'normal': 'green',
            'caution': 'yellow',
            'warning': 'orange',
            'critical': 'red',
            'emergency': 'darkred',
            'offline': 'gray'
        }

        for status, color in status_colors.items():
            status_sensors = [s for s in sensors if s['calculated_status'] == status]
            if status_sensors:
                sizes = [8 + s['severity_score'] * 10 for s in status_sensors]

                fig.add_trace(go.Scattermapbox(
                    lat=[s['lat'] for s in status_sensors],
                    lon=[s['lon'] for s in status_sensors],
                    mode='markers',
                    marker=dict(size=sizes, color=color, opacity=0.8),
                    text=[f"📡 {s['sensor_id']}<br>Status: {s['calculated_status']}<br>Reading: {s['current_reading']:.2f}<br>Water Depth: {s['water_depth']:.2f}m<br>Battery: {s['battery_level']:.0f}%<br>Signal: {s['signal_strength']:.0f}%"
                          for s in status_sensors],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'{status.title()} ({len(status_sensors)})'
                ))

    # Add flood zones as polygons/circles
    if flood_zones:
        for i, zone in enumerate(flood_zones):
            # Add zone center marker
            zone_color = 'darkred' if zone['zone_status'] == 'emergency' else 'red' if zone['zone_status'] == 'critical' else 'orange'

            fig.add_trace(go.Scattermapbox(
                lat=[zone['center_lat']],
                lon=[zone['center_lon']],
                mode='markers',
                marker=dict(size=25, color=zone_color, symbol='star', opacity=0.9),
                text=[f"🌊 FLOOD ZONE {i+1}<br>Status: {zone['zone_status']}<br>Area: {zone['area_km2']:.2f} km²<br>Sensors: {zone['sensor_count']}<br>Max Depth: {zone['max_water_depth']:.2f}m<br>Trend: {zone['trend']:.2f}"],
                hovertemplate='%{text}<extra></extra>',
                name=f"Zone {i+1} ({zone['zone_status']})"
            ))

            # Add zone boundary (approximate circle)
            radius_deg = np.sqrt(zone['area_km2'] / np.pi) / 111  # Convert km to degrees
            circle_lats = []
            circle_lons = []
            for angle in np.linspace(0, 2*np.pi, 50):
                circle_lats.append(zone['center_lat'] + radius_deg * np.cos(angle))
                circle_lons.append(zone['center_lon'] + radius_deg * np.sin(angle))

            fig.add_trace(go.Scattermapbox(
                lat=circle_lats,
                lon=circle_lons,
                mode='lines',
                line=dict(color=zone_color, width=3),
                showlegend=False,
                hoverinfo='skip'
            ))

    return fig

def main():
    """Main Streamlit application for sensor-based flood analysis with multi-agent integration"""
    st.set_page_config(page_title="Multi-Agent Sensor Flood Analyzer", layout="wide", page_icon="📡")

    st.title("📡 AI Multi-Agent Sensor-Based Flood Analysis System")
    st.markdown("**Real-time flood detection and analysis using IoT sensor network data with multi-agent coordination**")

    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SensorFloodAnalyzer()

    analyzer = st.session_state.analyzer

    # Sidebar controls
    st.sidebar.header("🔧 Analysis Configuration")

    # Connection status
    if analyzer.connected:
        st.sidebar.success("🟢 Redis Connected")
    else:
        st.sidebar.error("🔴 Redis Disconnected")
        if st.sidebar.button("🔄 Reconnect"):
            if analyzer.connect_to_redis():
                st.sidebar.success("✅ Reconnected!")
                st.rerun()

    # Multi-agent status
    st.sidebar.subheader("🤖 Multi-Agent Status")
    st.sidebar.info(f"Agent ID: {analyzer.agent_id}")
    st.sidebar.info(f"Alerts Sent: {len(analyzer.sent_alerts)}")
    st.sidebar.info(f"Cooldowns Active: {len(analyzer.alert_cooldown)}")

    # Analysis parameters
    max_sensors = st.sidebar.slider("Max sensors to analyze", 50, 1000, 500)
    analysis_hours = st.sidebar.slider("Analysis timeframe (hours)", 1, 72, 24)
    cluster_sensitivity = st.sidebar.slider("Zone detection sensitivity", 0.005, 0.02, 0.01, 0.001)
    min_zone_sensors = st.sidebar.slider("Min sensors per zone", 2, 10, 3)

    # Alert settings
    st.sidebar.subheader("🚨 Alert Configuration")
    min_severity = st.sidebar.slider("Min severity for alerts", 0.1, 1.0, 0.4, 0.1)
    cooldown_minutes = st.sidebar.slider("Alert cooldown (minutes)", 5, 60, 15, 5)

    # Update alert settings
    analyzer.alert_settings['min_severity_threshold'] = min_severity
    analyzer.alert_settings['cooldown_minutes'] = cooldown_minutes

    # Thresholds
    st.sidebar.subheader("🚨 Alert Thresholds")
    water_depth_critical = st.sidebar.slider("Critical water depth (m)", 1.0, 3.0, 1.5, 0.1)
    reading_critical = st.sidebar.slider("Critical sensor reading", 5.0, 15.0, 7.0, 0.5)

    # Update thresholds in analyzer
    analyzer.water_depth_thresholds['critical'] = water_depth_critical
    analyzer.reading_thresholds['critical'] = reading_critical

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", True)

    if st.sidebar.button("🔄 Run Analysis") or auto_refresh:
        with st.spinner("Fetching sensor network data..."):
            sensors = analyzer.fetch_sensor_data_with_timeframe(max_sensors, analysis_hours)

        if not analyzer.connected:
            st.error("❌ Cannot connect to Redis. Make sure Redis is running and sensor data is being generated.")
            st.stop()

        with st.spinner("Analyzing sensor network and sending alerts to coordination agent..."):
            analysis = analyzer.analyze_sensor_network(sensors)

        st.session_state.sensors = sensors
        st.session_state.analysis = analysis

    # Display results
    if hasattr(st.session_state, 'analysis'):
        analysis = st.session_state.analysis
        sensors = st.session_state.sensors

        # Risk level indicator
        risk_colors = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}
        network_risk = analysis['network_risk']

        st.markdown(f"## {risk_colors[network_risk['level']]} Network Risk Level: **{network_risk['level'].upper()}**")
        st.markdown(f"**Risk Score: {network_risk['score']:.1%} | Confidence: {network_risk['confidence']:.1%}**")

        # Multi-agent coordination status
        alerts_info = analysis.get('alerts_sent', {})
        col_alert1, col_alert2, col_alert3 = st.columns(3)
        with col_alert1:
            st.metric("Zone Alerts Sent", alerts_info.get('zones', 0))
        with col_alert2:
            st.metric("Critical Sensor Alerts", alerts_info.get('critical_sensors', 0))
        with col_alert3:
            st.metric("Total Alerts", alerts_info.get('total_alerts', 0))

        # Key metrics
        stats = analysis['network_stats']
        health = analysis['network_health']

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric("Total Sensors", stats['total_sensors'])
        with col2:
            st.metric("Active Sensors", stats['active_sensors'],
                      delta=stats['active_sensors'] - stats['total_sensors'])
        with col3:
            st.metric("Flooded Sensors", stats['flooded_sensors'])
        with col4:
            st.metric("Flood Zones", len(analysis['flood_zones']))
        with col5:
            st.metric("Max Water Depth", f"{stats['max_water_depth']:.2f}m")
        with col6:
            st.metric("Network Health", f"{health['network_coverage']:.1%}")

        # Main content areas
        col_map, col_details = st.columns([2, 1])

        with col_map:
            st.subheader("🗺️ Sensor Network & Flood Zones")

            # Calculate map center
            if sensors:
                center_lat = np.mean([s['lat'] for s in sensors])
                center_lon = np.mean([s['lon'] for s in sensors])
            else:
                center_lat, center_lon = 32.7767, -96.7970

            # Create and display map
            sensor_map = create_sensor_network_map(sensors, analysis['flood_zones'], center_lat, center_lon)
            st.plotly_chart(sensor_map, use_container_width=True)

        with col_details:
            st.subheader("📊 Multi-Agent Analysis")

            # Agent coordination status
            st.write("**🤖 Agent Coordination:**")
            st.write(f"📤 Alerts sent to coordination agent: {len(analyzer.sent_alerts)}")
            st.write(f"⏰ Active cooldowns: {len(analyzer.alert_cooldown)}")
            st.write(f"🎯 Alert threshold: {min_severity:.1%}")
            st.write(f"⏱️ Cooldown period: {cooldown_minutes} minutes")

            st.write("---")

            # Recommendations
            st.write("**🎯 Recommendations:**")
            for rec in analysis['recommendations']:
                st.write(f"• {rec}")

            st.write("---")

            # Network health indicators
            st.write("**🏥 Network Health:**")
            st.write(f"📊 Coverage: {health['network_coverage']:.1%}")
            st.write(f"🔋 Avg Battery: {health['avg_battery_level']:.1f}%")
            st.write(f"📡 Low Battery: {health['low_battery_sensors']} sensors")
            st.write(f"📶 Poor Signal: {health['poor_signal_sensors']} sensors")

            if health['avg_battery_level'] < 30:
                st.error("⚠️ Network battery levels critical!")
            elif health['avg_battery_level'] < 50:
                st.warning("🔋 Network battery levels low")
            else:
                st.success("✅ Network battery levels good")

            st.write("---")

            # Flood zone details
            if analysis['flood_zones']:
                st.write("**🌊 Flood Zone Details:**")
                for i, zone in enumerate(analysis['flood_zones'][:3]):
                    with st.expander(f"Zone {i+1} - {zone['zone_status'].title()}"):
                        st.write(f"📍 Center: {zone['center_lat']:.4f}, {zone['center_lon']:.4f}")
                        st.write(f"📏 Area: {zone['area_km2']:.2f} km²")
                        st.write(f"📡 Sensors: {zone['sensor_count']} ({zone['critical_sensors']} critical)")
                        st.write(f"💧 Max Depth: {zone['max_water_depth']:.2f}m")
                        st.write(f"📈 Trend: {zone['trend']:.2f} ({'worsening' if zone['trend'] > 0.1 else 'improving' if zone['trend'] < -0.1 else 'stable'})")
                        st.write(f"🎯 Confidence: {zone['confidence']:.1%}")

                        # Prediction info
                        pred = zone['prediction']
                        if pred['peak_expected']:
                            st.error("🚨 Peak conditions expected!")
                        elif pred['stabilizing']:
                            st.success("✅ Conditions stabilizing")
                        else:
                            st.info(f"📊 Next hour severity: {pred['next_hour_severity']:.1%}")

                        # Alert status
                        if zone['max_severity'] >= min_severity:
                            st.success("📤 Alert sent to coordination agent")
                        else:
                            st.info("📊 Below alert threshold")
            else:
                st.info("✅ No significant flood zones detected")

        # Detailed analysis tabs with multi-agent info
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "📡 Sensor Details", "🌊 Zone Analysis", "🔮 Predictions", "🤖 Agent Status"])

        with tab1:
            st.subheader("Sensor Network Trends")

            if sensors:
                # Create trend visualizations
                df_sensors = pd.DataFrame(sensors)

                # Status distribution pie chart
                status_counts = df_sensors['calculated_status'].value_counts()
                fig_pie = px.pie(values=status_counts.values, names=status_counts.index,
                                 title="Sensor Status Distribution",
                                 color_discrete_map={
                                     'normal': 'green', 'caution': 'yellow',
                                     'warning': 'orange', 'critical': 'red',
                                     'emergency': 'darkred'
                                 })
                st.plotly_chart(fig_pie, use_container_width=True)

                # Water depth vs reading scatter
                fig_scatter = px.scatter(df_sensors, x='current_reading', y='water_depth',
                                         color='calculated_status', size='severity_score',
                                         title="Sensor Reading vs Water Depth",
                                         color_discrete_map={
                                             'normal': 'green', 'caution': 'yellow',
                                             'warning': 'orange', 'critical': 'red',
                                             'emergency': 'darkred'
                                         })
                st.plotly_chart(fig_scatter, use_container_width=True)

        with tab2:
            st.subheader("Individual Sensor Analysis")

            if sensors:
                # Filter options
                col_filter1, col_filter2 = st.columns(2)
                with col_filter1:
                    status_filter = st.selectbox("Filter by status",
                                                 ['All'] + list(set([s['calculated_status'] for s in sensors])))
                with col_filter2:
                    sort_by = st.selectbox("Sort by", ['severity_score', 'water_depth', 'current_reading', 'battery_level'])

                # Filter and sort sensors
                filtered_sensors = sensors
                if status_filter != 'All':
                    filtered_sensors = [s for s in sensors if s['calculated_status'] == status_filter]

                filtered_sensors = sorted(filtered_sensors, key=lambda x: x[sort_by], reverse=True)

                # Display sensor table
                sensor_data = []
                for sensor in filtered_sensors[:20]:  # Show top 20
                    # Check if alert would be sent
                    alert_status = "📤 Alert sent" if sensor['severity_score'] >= min_severity and sensor['calculated_status'] in ['critical', 'emergency'] else "📊 Monitoring"

                    sensor_data.append({
                        'Sensor ID': sensor['sensor_id'],
                        'Status': sensor['calculated_status'],
                        'Severity': f"{sensor['severity_score']:.2f}",
                        'Water Depth (m)': f"{sensor['water_depth']:.2f}",
                        'Reading': f"{sensor['current_reading']:.2f}",
                        'Battery (%)': f"{sensor['battery_level']:.0f}",
                        'Signal (%)': f"{sensor['signal_strength']:.0f}",
                        'Flooded': '✅' if sensor['is_flooded'] else '❌',
                        'Agent Status': alert_status
                    })

                df_display = pd.DataFrame(sensor_data)
                st.dataframe(df_display, use_container_width=True)

        with tab3:
            st.subheader("Flood Zone Detailed Analysis")

            if analysis['flood_zones']:
                # Zone comparison chart with alert status
                zone_data = []
                for i, zone in enumerate(analysis['flood_zones']):
                    alert_sent = zone['max_severity'] >= min_severity
                    zone_data.append({
                        'Zone': f"Zone {i+1}",
                        'Status': zone['zone_status'],
                        'Area (km²)': zone['area_km2'],
                        'Sensors': zone['sensor_count'],
                        'Max Depth (m)': zone['max_water_depth'],
                        'Severity': zone['max_severity'],
                        'Trend': zone['trend'],
                        'Alert Sent': '✅' if alert_sent else '❌'
                    })

                df_zones = pd.DataFrame(zone_data)
                st.dataframe(df_zones, use_container_width=True)

        with tab4:
            st.subheader("Network Predictions & Forecasting")

            predictions = analysis['predictions']

            # Prediction metrics
            col_pred1, col_pred2, col_pred3 = st.columns(3)

            with col_pred1:
                st.metric("Next Hour Risk", f"{predictions['next_hour_risk']:.1%}")
            with col_pred2:
                st.metric("Peak Zones", predictions['peak_zones'])
            with col_pred3:
                st.metric("Expanding Zones", predictions['expanding_zones'])

            # Prediction details with agent coordination
            st.write("**🔮 Prediction Analysis:**")

            if predictions['next_hour_risk'] > 0.7:
                st.error(f"🚨 High risk predicted - agents will coordinate response")
            elif predictions['next_hour_risk'] > 0.4:
                st.warning(f"⚠️ Moderate risk predicted - enhanced monitoring")
            else:
                st.success(f"✅ Low risk predicted - routine monitoring")

        with tab5:
            st.subheader("🤖 Multi-Agent System Status")

            # Agent communication metrics
            col_agent1, col_agent2 = st.columns(2)

            with col_agent1:
                st.write("**📤 Outbound Communication:**")
                st.write(f"• Agent ID: `{analyzer.agent_id}`")
                st.write(f"• Total alerts sent: {len(analyzer.sent_alerts)}")
                st.write(f"• Zone alerts: {alerts_info.get('zones', 0)}")
                st.write(f"• Sensor alerts: {alerts_info.get('critical_sensors', 0)}")
                st.write(f"• Network updates sent: 1 per analysis")

            with col_agent2:
                st.write("**⏰ Alert Management:**")
                st.write(f"• Active cooldowns: {len(analyzer.alert_cooldown)}")
                st.write(f"• Cooldown period: {cooldown_minutes} minutes")
                st.write(f"• Min severity threshold: {min_severity:.1%}")
                st.write(f"• Last analysis: {analysis['timestamp'].strftime('%H:%M:%S')}")

            # Alert history
            if analyzer.sent_alerts:
                st.write("**📋 Recent Alert IDs:**")
                recent_alerts = list(analyzer.sent_alerts)[-10:]  # Show last 10
                for alert_id in recent_alerts:
                    st.code(alert_id)

            # Cooldown status
            if analyzer.alert_cooldown:
                st.write("**⏱️ Active Cooldowns:**")
                current_time = datetime.now()
                for location, last_time in analyzer.alert_cooldown.items():
                    remaining = timedelta(minutes=cooldown_minutes) - (current_time - last_time)
                    if remaining.total_seconds() > 0:
                        st.write(f"• {location}: {remaining.seconds // 60}m {remaining.seconds % 60}s remaining")

            # System health
            st.write("**🔧 System Health:**")
            st.write(f"• Redis connection: {'🟢 Connected' if analyzer.connected else '🔴 Disconnected'}")
            st.write(f"• Agent type: {analyzer.agent_type.value}")
            st.write(f"• Communication channels: Active")
            st.write(f"• Analysis frequency: {30 if auto_refresh else 'Manual'} seconds")

        # Last updated
        st.caption(f"Last analysis: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | Agent: {analyzer.agent_id}")

        # Auto-refresh
        if auto_refresh:
            time.sleep(30)
            st.rerun()

    else:
        st.info("👆 Click 'Run Analysis' or enable auto-refresh to start multi-agent sensor analysis")
        st.markdown("""
        ### 🚀 Multi-Agent Sensor-Based Flood Analysis System
        
        This system provides comprehensive flood analysis using IoT sensor network data with intelligent multi-agent coordination.
        
        #### 🤖 Multi-Agent Architecture:
        - **Sensor Analysis Agent**: Analyzes sensor data and detects flood zones
        - **Coordination Agent**: Receives alerts and coordinates response decisions
        - **Communication Agent**: Manages external communications and notifications
        - **Tweet Analysis Agent**: Processes social media data for correlation
        
        #### 📡 Agent Communication Features:
        - **Automatic Alert Generation**: Sends flood alerts to coordination agent
        - **Smart Cooldowns**: Prevents alert spam with time-based restrictions
        - **Severity Thresholds**: Configurable minimum severity for alert generation
        - **Zone-Based Alerts**: Intelligent grouping of sensors into flood zones
        - **Network Status Updates**: Broadcasts system health to other agents
        
        #### 🚨 Alert Types:
        1. **Flood Zone Alerts**: When clustered sensors detect flood conditions
        2. **Critical Sensor Alerts**: Individual sensors in emergency state
        3. **Network Status Updates**: Overall system health and predictions
        4. **Trend Alerts**: Zones showing concerning development patterns
        
        #### 🎯 Intelligence Features:
        - **Spatial Clustering**: Groups nearby sensors into coherent flood zones
        - **Temporal Analysis**: Tracks trends and predicts zone development
        - **Confidence Scoring**: Assesses reliability of detections
        - **Resource Planning**: Estimates affected areas for response coordination
        - **Predictive Analytics**: Forecasts flood development over time
        
        #### 📊 Configuration Options:
        - **Alert Thresholds**: Customize severity levels for different alert types
        - **Cooldown Periods**: Control frequency of alerts to prevent spam
        - **Zone Detection**: Adjust clustering sensitivity and minimum zone size
        - **Trend Analysis**: Configure temporal windows for pattern recognition
        - **Network Monitoring**: Set battery and signal strength thresholds
        
        #### 🔄 Real-Time Operation:
        1. **Continuous Monitoring**: Auto-refresh analyzes sensor data every 30 seconds
        2. **Intelligent Alerting**: Only sends alerts when conditions meet thresholds
        3. **Coordination**: Alerts trigger multi-agent response coordination
        4. **Adaptation**: System learns from patterns and adjusts predictions
        5. **Scalability**: Can handle hundreds of sensors across large areas
        
        #### 📋 Setup Requirements:
        1. **Redis Server**: Running on localhost with sensor data stream
        2. **Multi-Agent System**: Coordination and Communication agents active
        3. **Sensor Network**: IoT sensors publishing to 'sensor_data' stream
        4. **Network Coverage**: Adequate sensor deployment for zone detection
        5. **Agent Configuration**: Proper alert thresholds and cooldown settings
        
        ### 🎯 Key Benefits:
        - **Intelligent Coordination**: Avoids duplicate alerts and optimizes response
        - **Spatial Awareness**: Groups sensors into meaningful flood zones
        - **Predictive Capability**: Forecasts flood development and peak conditions
        - **Resource Optimization**: Provides area estimates for response planning
        - **Scalable Architecture**: Easily extensible with additional agent types
        - **Real-Time Response**: Immediate alerting for critical conditions
        """)

if __name__ == "__main__":
    main()