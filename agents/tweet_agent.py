import streamlit as st
import redis
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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

class TweetFloodPredictor:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        """Initialize the tweet-based flood predictor"""
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
        self.tweet_buffer = deque(maxlen=1000)  # Keep last 1000 tweets

        # Enhanced flood keywords for better prediction
        self.flood_keywords = {
            'emergency': ['trapped', 'help', 'emergency', 'rescue', 'urgent', '911', 'sos', 'stranded', 'evacuate'],
            'severe': ['catastrophic', 'devastating', 'massive flood', 'flash flood', 'dam breach', 'levee break'],
            'moderate': ['flooding', 'flooded', 'water rising', 'basement flood', 'road closed', 'evacuation', 'submerged'],
            'mild': ['wet roads', 'puddles', 'rain', 'soggy', 'drainage', 'storm', 'water accumulation'],
            'location_indicators': ['here', 'area', 'neighborhood', 'street', 'downtown', 'near', 'around']
        }

        # Prediction model weights
        self.prediction_weights = {
            'keyword_severity': 0.3,
            'sentiment_impact': 0.2,
            'temporal_clustering': 0.2,
            'spatial_clustering': 0.2,
            'tweet_velocity': 0.1
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
        """Enhanced tweet severity classification with prediction focus"""
        if not text:
            return 0.0

        text_lower = text.lower()
        severity_score = 0.0

        # Emergency keywords (highest weight)
        for keyword in self.flood_keywords['emergency']:
            if keyword in text_lower:
                severity_score += 1.0

        # Severe flood indicators
        for keyword in self.flood_keywords['severe']:
            if keyword in text_lower:
                severity_score += 0.8

        # Moderate flood indicators
        for keyword in self.flood_keywords['moderate']:
            if keyword in text_lower:
                severity_score += 0.6

        # Mild indicators
        for keyword in self.flood_keywords['mild']:
            if keyword in text_lower:
                severity_score += 0.3

        # Location specificity bonus (indicates local knowledge)
        for keyword in self.flood_keywords['location_indicators']:
            if keyword in text_lower:
                severity_score += 0.1

        # Enhanced sentiment analysis
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity

            # Negative sentiment increases severity for flood-related tweets
            if sentiment < -0.1:
                severity_score += abs(sentiment) * 0.5

            # Check for urgency markers
            urgency_markers = ['now', 'immediately', 'quickly', 'fast', 'asap']
            for marker in urgency_markers:
                if marker in text_lower:
                    severity_score += 0.2

        except:
            pass

        # Normalize to 0-1 scale
        return min(severity_score, 1.0)

    def fetch_tweets_with_timeframe(self, count=200, hours_back=24):
        """Fetch tweets from Redis stream within specified timeframe"""
        if not self.connected:
            return []

        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            start_timestamp = int(start_time.timestamp() * 1000)

            messages = self.redis_client.xrevrange(self.tweet_stream, count=count)
            tweets = []

            for message_id, fields in messages:
                # Parse timestamp from message ID or field
                try:
                    msg_timestamp = int(message_id.split('-')[0])
                    if msg_timestamp < start_timestamp:
                        continue
                except:
                    pass

                # Safely handle missing fields
                tweet = {
                    'id': message_id,
                    'user_id': fields.get('user_id', ''),
                    'username': fields.get('username', 'unknown'),
                    'text': fields.get('text', ''),
                    'lat': self._safe_float(fields.get('lat', 0)),
                    'lon': self._safe_float(fields.get('lon', 0)),
                    'timestamp': fields.get('timestamp', datetime.now().isoformat()),
                    'is_genuine': fields.get('is_genuine', 'False') == 'True',
                    'flood_severity': self._safe_float(fields.get('flood_severity', 0)),
                    'calculated_severity': self.classify_tweet_severity(fields.get('text', ''))
                }
                tweets.append(tweet)

            return tweets
        except Exception as e:
            st.error(f"Error fetching tweets: {e}")
            return []

    def _safe_float(self, value):
        """Safely convert value to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _safe_datetime(self, timestamp_str):
        """Safely parse datetime string"""
        if not timestamp_str:
            return datetime.now()

        try:
            # Handle multiple timestamp formats
            if 'Z' in timestamp_str:
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            elif '+' in timestamp_str or timestamp_str.endswith('00'):
                return datetime.fromisoformat(timestamp_str)
            else:
                # Assume ISO format without timezone
                return datetime.fromisoformat(timestamp_str)
        except ValueError:
            try:
                # Try parsing as timestamp
                return datetime.fromtimestamp(float(timestamp_str))
            except:
                return datetime.now()

    def detect_flood_hotspots(self, tweets, eps=0.01, min_samples=3):
        """Detect flood hotspots using spatial-temporal clustering"""
        if len(tweets) < min_samples:
            return []

        # Filter for flood-related tweets with higher threshold for prediction
        flood_tweets = [t for t in tweets if
                        t['is_genuine'] and
                        (t['flood_severity'] > 0.4 or t['calculated_severity'] > 0.5) and
                        t['lat'] != 0 and t['lon'] != 0]  # Valid coordinates

        if len(flood_tweets) < min_samples:
            return []

        # Prepare data for clustering
        coords = np.array([[t['lat'], t['lon']] for t in flood_tweets])
        severities = np.array([max(t['flood_severity'], t['calculated_severity']) for t in flood_tweets])

        # Validate coordinates
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            return []

        # Weight coordinates by severity for better clustering
        weighted_coords = coords * (1 + severities.reshape(-1, 1) * 2)

        # Standardize coordinates
        try:
            scaler = StandardScaler()
            coords_scaled = scaler.fit_transform(weighted_coords)
        except ValueError:
            return []

        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(coords_scaled)

        # Analyze clusters for prediction
        clusters = []
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise points
                continue

            cluster_tweets = [flood_tweets[i] for i in range(len(flood_tweets))
                              if cluster_labels[i] == cluster_id]

            if len(cluster_tweets) >= min_samples:
                lats = [t['lat'] for t in cluster_tweets]
                lons = [t['lon'] for t in cluster_tweets]
                severities = [max(t['flood_severity'], t['calculated_severity']) for t in cluster_tweets]

                # Calculate temporal spread safely
                timestamps = []
                for t in cluster_tweets:
                    try:
                        ts = self._safe_datetime(t['timestamp'])
                        timestamps.append(ts)
                    except:
                        continue

                if timestamps:
                    time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600  # hours
                else:
                    time_span = 0

                cluster_info = {
                    'id': f'hotspot_{cluster_id}',
                    'center_lat': np.mean(lats),
                    'center_lon': np.mean(lons),
                    'radius': np.sqrt(np.var(lats) + np.var(lons)) * 111,  # Rough km estimate
                    'tweet_count': len(cluster_tweets),
                    'avg_severity': np.mean(severities),
                    'max_severity': np.max(severities),
                    'time_span_hours': time_span,
                    'tweet_velocity': len(cluster_tweets) / max(time_span, 0.1),  # tweets per hour
                    'emergency_tweets': len([t for t in cluster_tweets if t['calculated_severity'] > 0.8]),
                    'confidence': min(len(cluster_tweets) / 8.0, 1.0),
                    'growth_trend': self.calculate_growth_trend(cluster_tweets),
                    'risk_prediction': self.predict_cluster_risk(cluster_tweets),
                    'sample_tweets': cluster_tweets[:5]
                }
                clusters.append(cluster_info)

        # Sort by predicted risk and confidence
        clusters.sort(key=lambda x: (x['risk_prediction'] * x['confidence']), reverse=True)
        return clusters

    def calculate_growth_trend(self, cluster_tweets):
        """Calculate if cluster is growing (positive trend indicates escalation)"""
        if len(cluster_tweets) < 3:
            return 0.0

        try:
            timestamps = []
            for tweet in cluster_tweets:
                try:
                    ts = self._safe_datetime(tweet['timestamp'])
                    timestamps.append(ts.timestamp())
                except:
                    continue

            if len(timestamps) < 3:
                return 0.0

            # Simple linear trend calculation
            timestamps.sort()
            x = np.arange(len(timestamps))
            y = np.array(timestamps)

            # Calculate slope (positive = growing over time)
            slope = np.polyfit(x, y, 1)[0]
            return min(max(slope / 3600, -1.0), 1.0)  # Normalize to -1 to 1

        except:
            return 0.0

    def predict_cluster_risk(self, cluster_tweets):
        """Predict risk level for a cluster using multiple factors"""
        if not cluster_tweets:
            return 0.0

        factors = {}

        # Factor 1: Keyword severity
        severities = [max(t['flood_severity'], t['calculated_severity']) for t in cluster_tweets]
        factors['keyword_severity'] = np.mean(severities)

        # Factor 2: Sentiment impact
        negative_sentiment_count = 0
        total_sentiment = 0
        for tweet in cluster_tweets:
            try:
                blob = TextBlob(tweet['text'])
                sentiment = blob.sentiment.polarity
                total_sentiment += abs(sentiment)
                if sentiment < -0.2:
                    negative_sentiment_count += 1
            except:
                continue

        factors['sentiment_impact'] = min(negative_sentiment_count / len(cluster_tweets), 1.0)

        # Factor 3: Temporal clustering (tweets happening close in time)
        timestamps = []
        for tweet in cluster_tweets:
            try:
                ts = self._safe_datetime(tweet['timestamp'])
                timestamps.append(ts.timestamp())
            except:
                continue

        if len(timestamps) > 1:
            time_variance = np.var(timestamps) / (3600 ** 2)  # Normalize by hour
            factors['temporal_clustering'] = max(0, 1 - time_variance)  # Lower variance = higher clustering
        else:
            factors['temporal_clustering'] = 0.5

        # Factor 4: Spatial clustering (tweets close geographically)
        lats = [t['lat'] for t in cluster_tweets if t['lat'] != 0]
        lons = [t['lon'] for t in cluster_tweets if t['lon'] != 0]

        if len(lats) > 1:
            spatial_variance = np.var(lats) + np.var(lons)
            factors['spatial_clustering'] = max(0, 1 - spatial_variance * 100)  # Adjust scale
        else:
            factors['spatial_clustering'] = 0.5

        # Factor 5: Tweet velocity (rapid increase indicates developing situation)
        if len(timestamps) > 2:
            time_span = (max(timestamps) - min(timestamps)) / 3600  # hours
            velocity = len(cluster_tweets) / max(time_span, 0.1)
            factors['tweet_velocity'] = min(velocity / 5.0, 1.0)  # Normalize
        else:
            factors['tweet_velocity'] = 0.3

        # Weighted prediction
        prediction = sum(factors[key] * self.prediction_weights[key]
                         for key in factors if key in self.prediction_weights)

        return min(prediction, 1.0)

    def predict_next_hour_risk(self, tweets):
        """Predict flood risk for the next hour based on trends"""
        if not tweets:
            return 0.0

        # Analyze tweet patterns in recent hours
        now = datetime.now()
        recent_tweets = []

        for tweet in tweets:
            try:
                tweet_time = self._safe_datetime(tweet['timestamp'])
                hours_ago = (now - tweet_time).total_seconds() / 3600
                if hours_ago <= 2:  # Last 2 hours
                    recent_tweets.append((tweet, hours_ago))
            except:
                continue

        if not recent_tweets:
            return 0.0

        # Calculate trend
        hour1_tweets = [t for t, h in recent_tweets if h <= 1]
        hour2_tweets = [t for t, h in recent_tweets if 1 < h <= 2]

        hour1_severity = np.mean([max(t['flood_severity'], t['calculated_severity'])
                                  for t in hour1_tweets]) if hour1_tweets else 0
        hour2_severity = np.mean([max(t['flood_severity'], t['calculated_severity'])
                                  for t in hour2_tweets]) if hour2_tweets else 0

        # Trend analysis
        if hour2_severity > 0:
            trend = (hour1_severity - hour2_severity) / hour2_severity
        else:
            trend = 0

        # Project next hour risk
        next_hour_risk = hour1_severity + (trend * hour1_severity * 0.5)
        return min(max(next_hour_risk, 0.0), 1.0)

    def predict_escalation(self, hotspots):
        """Predict probability of situation escalation"""
        if not hotspots:
            return 0.0

        escalation_factors = []

        for hotspot in hotspots:
            factors = [
                hotspot['tweet_velocity'] / 10.0,  # High velocity indicates rapid development
                hotspot['growth_trend'],           # Positive trend indicates escalation
                hotspot['emergency_tweets'] / max(hotspot['tweet_count'], 1),  # Emergency ratio
                min(hotspot['avg_severity'], 1.0) # Overall severity
            ]
            escalation_factors.extend(factors)

        return min(np.mean(escalation_factors), 1.0) if escalation_factors else 0.0

    def generate_predictions_recommendations(self, overall_risk, hotspots, emergency_tweets):
        """Generate actionable recommendations based on predictions"""
        recommendations = []

        if overall_risk > 0.75:
            recommendations.extend([
                "🚨 CRITICAL: Immediate emergency response recommended",
                "🚁 Deploy emergency teams to predicted hotspots",
                "📢 Issue immediate public flood warnings",
                "🚨 Monitor emergency tweets for rescue requests"
            ])
        elif overall_risk > 0.6:
            recommendations.extend([
                "⚠️ HIGH RISK: Prepare emergency resources",
                "📍 Focus monitoring on identified hotspots",
                "🚧 Consider preventive evacuations in high-risk areas",
                "📱 Increase social media monitoring frequency"
            ])
        elif overall_risk > 0.4:
            recommendations.extend([
                "🟡 MODERATE RISK: Enhanced monitoring recommended",
                "👀 Watch identified areas for escalation",
                "🏗️ Check drainage systems in potential hotspots",
                "📊 Continue trend analysis"
            ])
        else:
            recommendations.extend([
                "🟢 LOW RISK: Continue routine monitoring",
                "📈 Maintain baseline surveillance",
                "🔍 Monitor for emerging patterns"
            ])

        # Specific recommendations based on data
        if len(emergency_tweets) > 3:
            recommendations.append(f"🚨 {len(emergency_tweets)} emergency tweets detected - investigate immediately")

        if len(hotspots) > 2:
            recommendations.append(f"📍 {len(hotspots)} flood hotspots identified - prioritize resource allocation")

        # Growth trend warnings
        growing_hotspots = [h for h in hotspots if h['growth_trend'] > 0.3]
        if growing_hotspots:
            recommendations.append(f"📈 {len(growing_hotspots)} hotspots showing growth trends - expect escalation")

        return recommendations

    def predict_flood_risk(self, tweets):
        """Main flood prediction function using only tweets"""
        if not tweets:
            return {
                'overall_risk': 0.0,
                'risk_level': 'low',
                'confidence': 0.0,
                'hotspots': [],
                'total_tweets': 0,
                'genuine_tweets': 0,
                'flood_tweets': 0,
                'emergency_tweets': 0,
                'predictions': {
                    'next_hour_risk': 0.0,
                    'escalation_probability': 0.0,
                    'affected_areas': 0
                },
                'recommendations': ['🔍 No data available for prediction']
            }

        # Detect hotspots
        hotspots = self.detect_flood_hotspots(tweets)

        # Calculate overall metrics
        genuine_tweets = [t for t in tweets if t['is_genuine']]
        flood_tweets = [t for t in tweets if t['flood_severity'] > 0.3 or t['calculated_severity'] > 0.4]
        emergency_tweets = [t for t in tweets if t['calculated_severity'] > 0.8]

        # Overall risk calculation
        risk_factors = []

        if len(tweets) > 0:
            emergency_ratio = len(emergency_tweets) / len(tweets)
            flood_ratio = len(flood_tweets) / len(tweets)
            genuine_ratio = len(genuine_tweets) / len(tweets)

            risk_factors.extend([
                emergency_ratio * 2.0,  # Emergency tweets are critical
                flood_ratio * 1.5,      # Flood tweets are important
                genuine_ratio * 0.5     # Genuine tweets add credibility
            ])

        # Hotspot-based risk
        if hotspots:
            max_hotspot_risk = max([h['risk_prediction'] for h in hotspots])
            hotspot_density = len(hotspots) / 5.0  # Normalize
            risk_factors.extend([max_hotspot_risk, min(hotspot_density, 1.0)])

        # Calculate overall risk
        overall_risk = np.mean(risk_factors) if risk_factors else 0.0

        # Determine risk level
        if overall_risk > 0.75:
            risk_level = 'critical'
        elif overall_risk > 0.6:
            risk_level = 'high'
        elif overall_risk > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        # Calculate confidence based on data quality
        confidence = min(len(genuine_tweets) / 20.0, 1.0) if genuine_tweets else 0.0

        prediction_result = {
            'timestamp': datetime.now(),
            'overall_risk': overall_risk,
            'risk_level': risk_level,
            'confidence': confidence,
            'hotspots': hotspots,
            'total_tweets': len(tweets),
            'genuine_tweets': len(genuine_tweets),
            'flood_tweets': len(flood_tweets),
            'emergency_tweets': len(emergency_tweets),
            'predictions': {
                'next_hour_risk': self.predict_next_hour_risk(tweets),
                'escalation_probability': self.predict_escalation(hotspots),
                'affected_areas': len(hotspots)
            },
            'recommendations': self.generate_predictions_recommendations(overall_risk, hotspots, emergency_tweets)
        }

        return prediction_result

def main():
    """Main Streamlit application for tweet-based flood prediction"""
    st.set_page_config(page_title="Tweet Flood Predictor", layout="wide", page_icon="🌊")

    st.title("🌊 AI Tweet-Based Flood Predictor")
    st.markdown("**Real-time flood prediction using social media analysis**")

    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = TweetFloodPredictor()

    predictor = st.session_state.predictor

    # Sidebar controls
    st.sidebar.header("🔧 Prediction Settings")

    # Connection status
    if predictor.connected:
        st.sidebar.success("🟢 Redis Connected")
    else:
        st.sidebar.error("🔴 Redis Disconnected")
        if st.sidebar.button("🔄 Reconnect"):
            if predictor.connect_to_redis():
                st.sidebar.success("✅ Reconnected!")
                st.rerun()

    # Parameters
    max_tweets = st.sidebar.slider("Tweets to analyze", 100, 500, 250)
    prediction_hours = st.sidebar.slider("Prediction timeframe (hours)", 1, 48, 24)
    cluster_sensitivity = st.sidebar.slider("Hotspot sensitivity", 0.005, 0.02, 0.01, 0.001)
    min_cluster_size = st.sidebar.slider("Min hotspot size", 2, 8, 3)

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", False)  # Default to False for better performance

    if st.sidebar.button("🔄 Run Prediction") or auto_refresh:
        with st.spinner("Analyzing tweets and generating predictions..."):
            tweets = predictor.fetch_tweets_with_timeframe(max_tweets, prediction_hours)

        if not predictor.connected:
            st.error("❌ Cannot connect to Redis. Make sure Redis is running.")
            st.stop()

        with st.spinner("Generating flood predictions..."):
            prediction = predictor.predict_flood_risk(tweets)

        st.session_state.tweets = tweets
        st.session_state.prediction = prediction

    # Display predictions
    if hasattr(st.session_state, 'prediction'):
        prediction = st.session_state.prediction
        tweets = getattr(st.session_state, 'tweets', [])

        # Risk level indicator
        risk_colors = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}
        st.markdown(f"## {risk_colors[prediction['risk_level']]} Predicted Risk: **{prediction['risk_level'].upper()}**")
        st.markdown(f"**Confidence: {prediction['confidence']:.1%}**")

        # Key prediction metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Overall Risk", f"{prediction['overall_risk']:.1%}")
        with col2:
            st.metric("Next Hour Risk", f"{prediction['predictions']['next_hour_risk']:.1%}")
        with col3:
            st.metric("Escalation Probability", f"{prediction['predictions']['escalation_probability']:.1%}")
        with col4:
            st.metric("Predicted Hotspots", prediction['predictions']['affected_areas'])
        with col5:
            st.metric("Emergency Tweets", prediction['emergency_tweets'])

        # Main content
        col_map, col_pred = st.columns([2, 1])

        with col_map:
            st.subheader("🗺️ Predicted Flood Risk Map")

            # Create map focused on hotspots
            fig = go.Figure()

            if tweets and len(tweets) > 0:
                valid_coords = [(t['lat'], t['lon']) for t in tweets if t['lat'] != 0 and t['lon'] != 0]
                if valid_coords:
                    center_lat = np.mean([coord[0] for coord in valid_coords])
                    center_lon = np.mean([coord[1] for coord in valid_coords])
                else:
                    center_lat, center_lon = 32.7767, -96.7970  # Default to Dallas
            else:
                center_lat, center_lon = 32.7767, -96.7970

            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=11),
                height=500,
                margin={"r":0,"t":0,"l":0,"b":0}
            )

            # Add emergency tweets
            emergency_tweets = [t for t in tweets if t['calculated_severity'] > 0.8 and t['lat'] != 0 and t['lon'] != 0]
            if emergency_tweets:
                fig.add_trace(go.Scattermapbox(
                    lat=[t['lat'] for t in emergency_tweets],
                    lon=[t['lon'] for t in emergency_tweets],
                    mode='markers',
                    marker=dict(size=10, color='red', opacity=0.8),
                    text=[f"🚨 EMERGENCY<br>@{t['username']}<br>{t['text'][:80]}..." for t in emergency_tweets],
                    name=f'Emergency ({len(emergency_tweets)})'
                ))

            # Add predicted hotspots
            for i, hotspot in enumerate(prediction['hotspots']):
                size = 15 + hotspot['risk_prediction'] * 10
                color = 'darkred' if hotspot['risk_prediction'] > 0.7 else 'orange'

                fig.add_trace(go.Scattermapbox(
                    lat=[hotspot['center_lat']],
                    lon=[hotspot['center_lon']],
                    mode='markers',
                    marker=dict(size=size, color=color, symbol='star'),
                    text=[f"⭐ HOTSPOT {i+1}<br>Risk: {hotspot['risk_prediction']:.1%}<br>Tweets: {hotspot['tweet_count']}<br>Velocity: {hotspot['tweet_velocity']:.1f}/hr"],
                    name=f"Hotspot {i+1}"
                ))

            st.plotly_chart(fig, use_container_width=True)

        with col_pred:
            st.subheader("🔮 Predictions")

            # Recommendations
            st.write("**📋 Recommendations:**")
            for rec in prediction['recommendations']:
                st.write(f"• {rec}")

            st.write("---")

            # Hotspot predictions
            if prediction['hotspots']:
                st.write("**🎯 Hotspot Analysis:**")
                for i, hotspot in enumerate(prediction['hotspots'][:3]):
                    with st.expander(f"Hotspot {i+1} - Risk: {hotspot['risk_prediction']:.1%}"):
                        st.write(f"📍 Location: {hotspot['center_lat']:.4f}, {hotspot['center_lon']:.4f}")
                        st.write(f"📊 Risk Score: {hotspot['risk_prediction']:.1%}")
                        st.write(f"🚀 Tweet Velocity: {hotspot['tweet_velocity']:.1f}/hour")
                        st.write(f"📈 Growth Trend: {hotspot['growth_trend']:.2f}")
                        st.write(f"🎯 Confidence: {hotspot['confidence']:.1%}")

                        if hotspot['growth_trend'] > 0.3:
                            st.warning("⚠️ Growing trend detected!")

                        # Sample tweets
                        if hotspot['sample_tweets']:
                            st.write("**Sample tweets:**")
                            for tweet in hotspot['sample_tweets'][:2]:
                                st.caption(f"@{tweet['username']}: {tweet['text'][:80]}...")
            else:
                st.info("No significant hotspots predicted")

        # Detailed analysis tabs
        tab1, tab2, tab3 = st.tabs(["📈 Trends", "🔍 Tweet Analysis", "📊 Model Details"])

        with tab1:
            st.subheader("Prediction Trends")

            if tweets:
                try:
                    # Risk over time
                    df_tweets = pd.DataFrame(tweets)

                    # Clean timestamp data
                    valid_timestamps = []
                    for idx, tweet in enumerate(tweets):
                        try:
                            ts = predictor._safe_datetime(tweet['timestamp'])
                            valid_timestamps.append((idx, ts))
                        except:
                            continue

                    if valid_timestamps:
                        # Create DataFrame with valid timestamps
                        valid_indices = [idx for idx, _ in valid_timestamps]
                        timestamps = [ts for _, ts in valid_timestamps]

                        risk_scores = [max(tweets[idx]['flood_severity'], tweets[idx]['calculated_severity'])
                                       for idx in valid_indices]

                        trend_df = pd.DataFrame({
                            'timestamp': timestamps,
                            'risk_score': risk_scores
                        })

                        trend_df['hour'] = trend_df['timestamp'].dt.floor('H')

                        trend_data = trend_df.groupby('hour').agg({
                            'risk_score': 'mean',
                            'timestamp': 'count'
                        }).reset_index()
                        trend_data.columns = ['hour', 'avg_risk', 'tweet_count']

                        if len(trend_data) > 1:
                            fig_trend = px.line(trend_data, x='hour', y='avg_risk',
                                                title="Risk Score Trend Over Time")
                            st.plotly_chart(fig_trend, use_container_width=True)
                        else:
                            st.info("Not enough temporal data for trend analysis")
                    else:
                        st.warning("No valid timestamps found in tweet data")
                except Exception as e:
                    st.error(f"Error generating trend analysis: {e}")
            else:
                st.info("No tweet data available for trend analysis")

        with tab2:
            st.subheader("Tweet Content Analysis")

            if tweets:
                try:
                    # Severity distribution
                    severities = [max(t['flood_severity'], t['calculated_severity']) for t in tweets]
                    if severities:
                        fig_hist = px.histogram(x=severities, nbins=20,
                                                title="Tweet Severity Distribution",
                                                labels={'x': 'Severity Score', 'y': 'Count'})
                        st.plotly_chart(fig_hist, use_container_width=True)

                    # High-risk tweets
                    high_risk_tweets = [t for t in tweets if max(t['flood_severity'], t['calculated_severity']) > 0.7]
                    if high_risk_tweets:
                        st.write("**⚠️ High-Risk Tweets:**")
                        for tweet in high_risk_tweets[:5]:
                            risk_score = max(tweet['flood_severity'], tweet['calculated_severity'])
                            st.warning(f"Risk: {risk_score:.1%} | @{tweet['username']}: {tweet['text'][:100]}...")
                    else:
                        st.info("No high-risk tweets detected")

                    # Tweet statistics
                    st.write("**📊 Tweet Statistics:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Tweets", len(tweets))
                    with col2:
                        genuine_count = len([t for t in tweets if t['is_genuine']])
                        st.metric("Genuine Tweets", genuine_count)
                    with col3:
                        avg_severity = np.mean(severities) if severities else 0
                        st.metric("Avg Severity", f"{avg_severity:.1%}")

                except Exception as e:
                    st.error(f"Error in tweet analysis: {e}")
            else:
                st.info("No tweet data available for analysis")

        with tab3:
            st.subheader("Prediction Model Details")

            st.write("**🔧 Model Weights:**")
            for factor, weight in predictor.prediction_weights.items():
                st.write(f"• {factor.replace('_', ' ').title()}: {weight:.1%}")

            st.write("**📊 Current Analysis:**")
            st.write(f"• Total tweets analyzed: {prediction['total_tweets']}")
            if prediction['total_tweets'] > 0:
                st.write(f"• Genuine tweets: {prediction['genuine_tweets']} ({prediction['genuine_tweets']/prediction['total_tweets']:.1%})")
            else:
                st.write("• Genuine tweets: 0 (0.0%)")
            st.write(f"• Flood-related tweets: {prediction['flood_tweets']}")
            st.write(f"• Emergency tweets: {prediction['emergency_tweets']}")

            # Model parameters
            st.write("**⚙️ Detection Parameters:**")
            st.write(f"• Cluster sensitivity: {cluster_sensitivity}")
            st.write(f"• Minimum cluster size: {min_cluster_size}")
            st.write(f"• Prediction timeframe: {prediction_hours} hours")

    else:
        st.info("Click 'Run Prediction' to analyze tweets and generate flood predictions")

        # Show sample data structure for debugging
        st.subheader("🔧 System Status")
        if predictor.connected:
            st.success("✅ Redis connection active")
        else:
            st.error("❌ Redis connection failed")
            st.write("**Troubleshooting:**")
            st.write("1. Ensure Redis server is running")
            st.write("2. Check Redis connection parameters")
            st.write("3. Verify tweet stream exists")

    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()