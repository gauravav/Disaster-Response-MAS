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
import re
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class FloodTweetAnalyzer:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        """Initialize the flood tweet analyzer"""
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

        self.tweet_stream = "flood_tweets"
        self.sensor_stream = "sensor_data"

        # Data storage
        self.tweet_buffer = deque(maxlen=1000)  # Keep last 1000 tweets
        self.sensor_buffer = deque(maxlen=500)  # Keep last 500 sensor readings
        self.flood_hotspots = []
        self.last_analysis_time = datetime.now()

        # Flood keywords for analysis
        self.flood_keywords = {
            'severe': ['trapped', 'help', 'emergency', 'rescue', 'urgent', 'catastrophic', 'evacuate', 'stranded'],
            'moderate': ['flooding', 'flooded', 'water rising', 'basement flood', 'road closed', 'evacuation'],
            'mild': ['wet roads', 'puddles', 'rain', 'soggy', 'drainage', 'storm'],
            'emergency': ['911', 'help', 'sos', 'trapped', 'rescue', 'emergency', 'urgent']
        }

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

    def classify_tweet_severity(self, text):
        """Classify tweet severity based on keywords and sentiment"""
        if not text:
            return 0.0

        text_lower = text.lower()
        severity_score = 0.0

        # Keyword-based scoring
        for keyword in self.flood_keywords['emergency']:
            if keyword in text_lower:
                severity_score += 1.0

        for keyword in self.flood_keywords['severe']:
            if keyword in text_lower:
                severity_score += 0.8

        for keyword in self.flood_keywords['moderate']:
            if keyword in text_lower:
                severity_score += 0.6

        for keyword in self.flood_keywords['mild']:
            if keyword in text_lower:
                severity_score += 0.3

        # Sentiment analysis
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            # Negative sentiment increases severity for flood-related tweets
            if sentiment < -0.1:
                severity_score += abs(sentiment) * 0.5
        except:
            pass

        # Normalize to 0-1 scale
        return min(severity_score, 1.0)

    def fetch_latest_tweets(self, count=100):
        """Fetch latest tweets from Redis stream"""
        if not self.connected:
            return []

        try:
            messages = self.redis_client.xrevrange(self.tweet_stream, count=count)
            tweets = []

            for message_id, fields in messages:
                tweet = {
                    'id': message_id,
                    'user_id': fields.get('user_id', ''),
                    'username': fields.get('username', ''),
                    'text': fields.get('text', ''),
                    'lat': float(fields.get('lat', 0)),
                    'lon': float(fields.get('lon', 0)),
                    'timestamp': fields.get('timestamp', ''),
                    'is_genuine': fields.get('is_genuine', 'False') == 'True',
                    'flood_severity': float(fields.get('flood_severity', 0)),
                    'calculated_severity': self.classify_tweet_severity(fields.get('text', ''))
                }
                tweets.append(tweet)

            return tweets
        except Exception as e:
            st.error(f"Error fetching tweets: {e}")
            return []

    def fetch_latest_sensors(self, count=50):
        """Fetch latest sensor data from Redis stream"""
        if not self.connected:
            return []

        try:
            messages = self.redis_client.xrevrange(self.sensor_stream, count=count)
            sensors = []

            for message_id, fields in messages:
                sensor = {
                    'id': message_id,
                    'sensor_id': fields.get('sensor_id', ''),
                    'lat': float(fields.get('lat', 0)),
                    'lon': float(fields.get('lon', 0)),
                    'current_reading': float(fields.get('current_reading', 0)),
                    'water_depth': float(fields.get('water_depth', 0)),
                    'alert_level': fields.get('alert_level', 'normal'),
                    'is_flooded': fields.get('is_flooded', 'False') == 'True',
                    'timestamp': fields.get('timestamp', '')
                }
                sensors.append(sensor)

            return sensors
        except Exception as e:
            st.error(f"Error fetching sensors: {e}")
            return []

    def detect_flood_clusters(self, tweets, eps=0.01, min_samples=3):
        """Use DBSCAN clustering to identify flood hotspots from tweets"""
        if len(tweets) < min_samples:
            return []

        # Filter for flood-related tweets
        flood_tweets = [t for t in tweets if
                        t['is_genuine'] and
                        (t['flood_severity'] > 0.3 or t['calculated_severity'] > 0.4)]

        if len(flood_tweets) < min_samples:
            return []

        # Prepare data for clustering
        coords = np.array([[t['lat'], t['lon']] for t in flood_tweets])
        severities = np.array([max(t['flood_severity'], t['calculated_severity']) for t in flood_tweets])

        # Weight coordinates by severity
        weighted_coords = coords * (1 + severities.reshape(-1, 1))

        # Standardize coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(weighted_coords)

        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(coords_scaled)

        # Analyze clusters
        clusters = []
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise points
                continue

            cluster_tweets = [flood_tweets[i] for i in range(len(flood_tweets))
                              if cluster_labels[i] == cluster_id]

            if len(cluster_tweets) >= min_samples:
                # Calculate cluster statistics
                lats = [t['lat'] for t in cluster_tweets]
                lons = [t['lon'] for t in cluster_tweets]
                severities = [max(t['flood_severity'], t['calculated_severity']) for t in cluster_tweets]

                cluster_info = {
                    'id': f'cluster_{cluster_id}',
                    'center_lat': np.mean(lats),
                    'center_lon': np.mean(lons),
                    'radius': np.std(lats) + np.std(lons),  # Rough radius estimate
                    'tweet_count': len(cluster_tweets),
                    'avg_severity': np.mean(severities),
                    'max_severity': np.max(severities),
                    'genuine_tweets': len([t for t in cluster_tweets if t['is_genuine']]),
                    'emergency_tweets': len([t for t in cluster_tweets if t['calculated_severity'] > 0.8]),
                    'confidence': min(len(cluster_tweets) / 10.0, 1.0),  # Confidence based on tweet density
                    'tweets': cluster_tweets[:10]  # Keep sample tweets
                }
                clusters.append(cluster_info)

        # Sort by confidence and severity
        clusters.sort(key=lambda x: (x['confidence'] * x['avg_severity']), reverse=True)
        return clusters

    def analyze_flood_risk(self, tweets, sensors):
        """Comprehensive flood risk analysis combining tweets and sensors"""
        analysis = {
            'timestamp': datetime.now(),
            'total_tweets': len(tweets),
            'genuine_tweets': len([t for t in tweets if t['is_genuine']]),
            'flood_tweets': len([t for t in tweets if t['flood_severity'] > 0.3]),
            'emergency_tweets': len([t for t in tweets if t['calculated_severity'] > 0.8]),
            'active_sensors': len([s for s in sensors if s['alert_level'] != 'offline']),
            'flooded_sensors': len([s for s in sensors if s['is_flooded']]),
            'critical_sensors': len([s for s in sensors if s['alert_level'] == 'critical']),
            'clusters': [],
            'risk_level': 'low',
            'recommendations': []
        }

        # Detect flood clusters
        clusters = self.detect_flood_clusters(tweets)
        analysis['clusters'] = clusters

        # Calculate overall risk level
        risk_factors = []

        # Tweet-based risk factors
        if analysis['total_tweets'] > 0:
            emergency_ratio = analysis['emergency_tweets'] / analysis['total_tweets']
            flood_ratio = analysis['flood_tweets'] / analysis['total_tweets']
            risk_factors.extend([emergency_ratio * 2, flood_ratio])

        # Sensor-based risk factors
        if analysis['active_sensors'] > 0:
            flooded_ratio = analysis['flooded_sensors'] / analysis['active_sensors']
            critical_ratio = analysis['critical_sensors'] / analysis['active_sensors']
            risk_factors.extend([flooded_ratio, critical_ratio])

        # Cluster-based risk factors
        if clusters:
            max_cluster_severity = max([c['avg_severity'] for c in clusters])
            cluster_density = len(clusters) / 10.0  # Normalize
            risk_factors.extend([max_cluster_severity, min(cluster_density, 1.0)])

        # Calculate overall risk
        if risk_factors:
            overall_risk = np.mean(risk_factors)
            if overall_risk > 0.7:
                analysis['risk_level'] = 'critical'
            elif overall_risk > 0.5:
                analysis['risk_level'] = 'high'
            elif overall_risk > 0.3:
                analysis['risk_level'] = 'medium'
            else:
                analysis['risk_level'] = 'low'

        # Generate recommendations
        analysis['recommendations'] = self.generate_recommendations(analysis)

        return analysis

    def generate_recommendations(self, analysis):
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        if analysis['risk_level'] == 'critical':
            recommendations.extend([
                "🚨 IMMEDIATE ACTION REQUIRED",
                "Deploy emergency response teams to identified hotspots",
                "Activate evacuation procedures for high-risk areas",
                "Monitor social media for rescue requests"
            ])

        elif analysis['risk_level'] == 'high':
            recommendations.extend([
                "⚠️ High flood risk detected",
                "Increase monitoring of identified clusters",
                "Prepare emergency resources",
                "Issue public flood warnings"
            ])

        elif analysis['risk_level'] == 'medium':
            recommendations.extend([
                "🟡 Moderate flood activity",
                "Continue monitoring affected areas",
                "Check drainage systems in hotspots",
                "Inform local authorities"
            ])

        if analysis['emergency_tweets'] > 5:
            recommendations.append(f"📱 {analysis['emergency_tweets']} emergency tweets detected - investigate immediately")

        if analysis['clusters']:
            recommendations.append(f"📍 {len(analysis['clusters'])} flood hotspots identified - prioritize these areas")

        if analysis['critical_sensors'] > 0:
            recommendations.append(f"📡 {analysis['critical_sensors']} sensors in critical state - verify conditions")

        return recommendations

def create_flood_map(tweets, sensors, clusters, center_lat=32.7767, center_lon=-96.7970):
    """Create interactive flood risk map"""
    fig = go.Figure()

    # Add base map
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11
        ),
        height=600,
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    # Add tweet markers
    if tweets:
        # Separate tweets by type
        genuine_flood_tweets = [t for t in tweets if t['is_genuine'] and t['flood_severity'] > 0.3]
        emergency_tweets = [t for t in tweets if t['calculated_severity'] > 0.8]
        normal_tweets = [t for t in tweets if t['flood_severity'] <= 0.3 and t['calculated_severity'] <= 0.4]

        # Emergency tweets (red)
        if emergency_tweets:
            fig.add_trace(go.Scattermapbox(
                lat=[t['lat'] for t in emergency_tweets],
                lon=[t['lon'] for t in emergency_tweets],
                mode='markers',
                marker=dict(size=12, color='red', opacity=0.8),
                text=[f"🚨 EMERGENCY<br>@{t['username']}<br>{t['text'][:100]}..." for t in emergency_tweets],
                hovertemplate='%{text}<extra></extra>',
                name=f'Emergency Tweets ({len(emergency_tweets)})'
            ))

        # Flood tweets (orange)
        if genuine_flood_tweets:
            fig.add_trace(go.Scattermapbox(
                lat=[t['lat'] for t in genuine_flood_tweets],
                lon=[t['lon'] for t in genuine_flood_tweets],
                mode='markers',
                marker=dict(size=8, color='orange', opacity=0.6),
                text=[f"🌊 FLOOD<br>@{t['username']}<br>{t['text'][:100]}..." for t in genuine_flood_tweets],
                hovertemplate='%{text}<extra></extra>',
                name=f'Flood Tweets ({len(genuine_flood_tweets)})'
            ))

        # Normal tweets (blue)
        if normal_tweets and len(normal_tweets) <= 50:  # Limit normal tweets for clarity
            fig.add_trace(go.Scattermapbox(
                lat=[t['lat'] for t in normal_tweets[:50]],
                lon=[t['lon'] for t in normal_tweets[:50]],
                mode='markers',
                marker=dict(size=4, color='lightblue', opacity=0.3),
                text=[f"💬 @{t['username']}<br>{t['text'][:100]}..." for t in normal_tweets[:50]],
                hovertemplate='%{text}<extra></extra>',
                name=f'Normal Tweets ({len(normal_tweets)})'
            ))

    # Add sensor markers
    if sensors:
        sensor_colors = {
            'critical': 'darkred',
            'warning': 'darkorange',
            'caution': 'gold',
            'normal': 'green'
        }

        for alert_level, color in sensor_colors.items():
            level_sensors = [s for s in sensors if s['alert_level'] == alert_level]
            if level_sensors:
                fig.add_trace(go.Scattermapbox(
                    lat=[s['lat'] for s in level_sensors],
                    lon=[s['lon'] for s in level_sensors],
                    mode='markers',
                    marker=dict(size=10, color=color, symbol='circle'),
                    text=[f"📡 {s['sensor_id']}<br>Reading: {s['current_reading']:.2f}m<br>Depth: {s['water_depth']:.1f}m"
                          for s in level_sensors],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'{alert_level.title()} Sensors ({len(level_sensors)})'
                ))

    # Add flood clusters as circles
    if clusters:
        for i, cluster in enumerate(clusters):
            # Add cluster center
            fig.add_trace(go.Scattermapbox(
                lat=[cluster['center_lat']],
                lon=[cluster['center_lon']],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                text=[f"🎯 HOTSPOT {i+1}<br>Tweets: {cluster['tweet_count']}<br>Severity: {cluster['avg_severity']:.2f}<br>Confidence: {cluster['confidence']:.2f}"],
                hovertemplate='%{text}<extra></extra>',
                name=f"Hotspot {i+1}"
            ))

    return fig

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="Flood Tracking Agent", layout="wide", page_icon="🌊")

    st.title("🌊 AI Flood Tracking Agent")
    st.markdown("**Real-time flood detection and risk analysis from social media and sensor data**")

    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = FloodTweetAnalyzer()

    analyzer = st.session_state.analyzer

    # Sidebar controls
    st.sidebar.header("🔧 Configuration")

    # Redis connection status
    if analyzer.connected:
        st.sidebar.success("🟢 Redis Connected")
    else:
        st.sidebar.error("🔴 Redis Disconnected")
        if st.sidebar.button("🔄 Reconnect"):
            if analyzer.connect_to_redis():
                st.sidebar.success("✅ Reconnected!")
                st.rerun()

    # Analysis parameters
    max_tweets = st.sidebar.slider("Max tweets to analyze", 50, 500, 200)
    max_sensors = st.sidebar.slider("Max sensors to show", 10, 100, 50)
    cluster_sensitivity = st.sidebar.slider("Cluster sensitivity", 0.005, 0.02, 0.01, 0.001)
    min_cluster_size = st.sidebar.slider("Min cluster size", 2, 10, 3)

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", True)

    if st.sidebar.button("🔄 Manual Refresh") or auto_refresh:
        # Fetch latest data
        with st.spinner("Fetching latest data..."):
            tweets = analyzer.fetch_latest_tweets(max_tweets)
            sensors = analyzer.fetch_latest_sensors(max_sensors)

        if not analyzer.connected:
            st.error("❌ Cannot connect to Redis. Make sure Redis is running and the flood simulation is generating data.")
            st.stop()

        # Perform analysis
        with st.spinner("Analyzing flood risk..."):
            analysis = analyzer.analyze_flood_risk(tweets, sensors)

        # Store in session state
        st.session_state.tweets = tweets
        st.session_state.sensors = sensors
        st.session_state.analysis = analysis

    # Display results if we have data
    if hasattr(st.session_state, 'analysis'):
        analysis = st.session_state.analysis
        tweets = st.session_state.tweets
        sensors = st.session_state.sensors

        # Risk level indicator
        risk_colors = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🟠',
            'critical': '🔴'
        }

        st.markdown(f"## {risk_colors[analysis['risk_level']]} Current Risk Level: **{analysis['risk_level'].upper()}**")

        # Key metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric("Total Tweets", analysis['total_tweets'])
        with col2:
            st.metric("Emergency Tweets", analysis['emergency_tweets'])
        with col3:
            st.metric("Flood Hotspots", len(analysis['clusters']))
        with col4:
            st.metric("Active Sensors", analysis['active_sensors'])
        with col5:
            st.metric("Critical Sensors", analysis['critical_sensors'])
        with col6:
            st.metric("Flooded Sensors", analysis['flooded_sensors'])

        # Main content areas
        col_map, col_details = st.columns([2, 1])

        with col_map:
            st.subheader("🗺️ Flood Risk Map")

            # Calculate map center from data
            if tweets:
                center_lat = np.mean([t['lat'] for t in tweets])
                center_lon = np.mean([t['lon'] for t in tweets])
            else:
                center_lat, center_lon = 32.7767, -96.7970  # Default to Dallas

            # Create and display map
            flood_map = create_flood_map(tweets, sensors, analysis['clusters'], center_lat, center_lon)
            st.plotly_chart(flood_map, use_container_width=True)

        with col_details:
            st.subheader("📊 Analysis Details")

            # Recommendations
            st.write("**🎯 Recommendations:**")
            for rec in analysis['recommendations']:
                st.write(f"• {rec}")

            st.write("---")

            # Hotspot details
            if analysis['clusters']:
                st.write("**🔥 Flood Hotspots:**")
                for i, cluster in enumerate(analysis['clusters'][:5]):  # Show top 5
                    with st.expander(f"Hotspot {i+1} - Severity: {cluster['avg_severity']:.2f}"):
                        st.write(f"📍 Location: {cluster['center_lat']:.4f}, {cluster['center_lon']:.4f}")
                        st.write(f"📱 Tweets: {cluster['tweet_count']} ({cluster['genuine_tweets']} genuine)")
                        st.write(f"🚨 Emergency: {cluster['emergency_tweets']} tweets")
                        st.write(f"🎯 Confidence: {cluster['confidence']:.2f}")

                        # Sample tweets
                        st.write("**Sample Tweets:**")
                        for tweet in cluster['tweets'][:3]:
                            st.caption(f"@{tweet['username']}: {tweet['text'][:100]}...")
            else:
                st.info("No significant flood hotspots detected")

        # Additional analysis tabs
        tab1, tab2, tab3 = st.tabs(["📈 Trends", "📱 Tweet Analysis", "📡 Sensor Status"])

        with tab1:
            st.subheader("Flood Activity Trends")

            if tweets:
                # Create timeline chart
                df_tweets = pd.DataFrame(tweets)
                df_tweets['timestamp'] = pd.to_datetime(df_tweets['timestamp'])
                df_tweets['hour'] = df_tweets['timestamp'].dt.floor('H')

                # Group by hour and severity
                trend_data = df_tweets.groupby(['hour', 'is_genuine']).size().reset_index(name='count')

                fig_trend = px.line(trend_data, x='hour', y='count', color='is_genuine',
                                    title="Tweet Volume Over Time")
                st.plotly_chart(fig_trend, use_container_width=True)

        with tab2:
            st.subheader("Tweet Content Analysis")

            if tweets:
                # Severity distribution
                severities = [max(t['flood_severity'], t['calculated_severity']) for t in tweets]
                fig_hist = px.histogram(x=severities, nbins=20, title="Tweet Severity Distribution")
                st.plotly_chart(fig_hist, use_container_width=True)

                # Recent emergency tweets
                emergency_tweets = [t for t in tweets if t['calculated_severity'] > 0.8]
                if emergency_tweets:
                    st.write("**🚨 Recent Emergency Tweets:**")
                    for tweet in emergency_tweets[:10]:
                        st.error(f"@{tweet['username']}: {tweet['text']}")

        with tab3:
            st.subheader("Sensor Network Status")

            if sensors:
                # Sensor status pie chart
                status_counts = pd.Series([s['alert_level'] for s in sensors]).value_counts()
                fig_pie = px.pie(values=status_counts.values, names=status_counts.index,
                                 title="Sensor Alert Level Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)

                # Sensor data table
                sensor_df = pd.DataFrame(sensors)
                st.dataframe(sensor_df[['sensor_id', 'current_reading', 'water_depth', 'alert_level', 'is_flooded']],
                             use_container_width=True)

        # Last updated
        st.caption(f"Last updated: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

        # Auto-refresh mechanism
        if auto_refresh:
            time.sleep(30)
            st.rerun()

    else:
        st.info("👆 Click 'Manual Refresh' or enable auto-refresh to start analyzing flood data")
        st.markdown("""
        ### 🚀 Getting Started:
        1. **Make sure Redis is running**: `redis-server`
        2. **Start the flood simulation** in another terminal
        3. **Enable auto-refresh** or click manual refresh
        4. **Monitor the map** for flood hotspots and emergency tweets
        
        ### 🎯 Features:
        - **Real-time tweet analysis** with AI sentiment classification
        - **Flood hotspot detection** using machine learning clustering
        - **Interactive risk map** with tweets, sensors, and hotspots
        - **Emergency alert system** for critical situations
        - **Actionable recommendations** for flood response
        """)

if __name__ == "__main__":
    main()