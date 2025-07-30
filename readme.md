# Multi-Agent Flood Rescue System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Installation and Setup](#installation-and-setup)
4. [Agent Documentation](#agent-documentation)
5. [Data Structures](#data-structures)
6. [Message Protocols](#message-protocols)
7. [API Reference](#api-reference)
8. [Configuration](#configuration)
9. [Deployment Guide](#deployment-guide)
10. [Troubleshooting](#troubleshooting)
11. [Performance Guidelines](#performance-guidelines)
12. [Extension Guide](#extension-guide)

---

## System Overview

### Purpose
The Multi-Agent Flood Rescue System is an intelligent disaster response platform that combines IoT sensor data and social media analysis to provide real-time flood detection, coordinated emergency response, and automated communication with emergency services and the public.

### Key Features
- **Real-time flood detection** from multiple data sources
- **Intelligent spatial clustering** to avoid duplicate responses
- **Automated resource allocation** and priority management
- **Multi-channel communication** to emergency services and public
- **Predictive analytics** for flood development forecasting
- **Scalable agent-based architecture** for easy expansion

### System Components
- **Sensor Analysis Agent**: Processes IoT sensor data for flood detection
- **Tweet Analysis Agent**: Analyzes social media for flood-related content
- **Coordination Agent**: Makes intelligent decisions and resource allocation
- **Communication Agent**: Manages external communications and notifications
- **Redis Backend**: Handles message passing and data storage

---

## Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌─────────────────┐
│   Flood Data    │    │  Social Media   │
│   Simulation    │    │     Data        │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Sensor Analysis │    │ Tweet Analysis  │
│     Agent       │    │     Agent       │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     ▼
           ┌─────────────────┐
           │  Coordination   │
           │     Agent       │
           └─────────┬───────┘
                     ▼
           ┌─────────────────┐
           │ Communication   │
           │     Agent       │
           └─────────┬───────┘
                     ▼
    ┌────────────────┬────────────────┐
    ▼                ▼                ▼
┌───────────┐ ┌──────────────┐ ┌──────────┐
│Emergency  │ │    Public    │ │  Other   │
│ Services  │ │Notifications │ │  Agents  │
└───────────┘ └──────────────┘ └──────────┘
```

### Agent Communication Flow
```
Data Source → Analysis Agent → Coordination Agent → Communication Agent → External Systems
     ↓              ↓                ↓                    ↓               ↓
   Redis          Redis            Redis               Redis          External
  Streams        Pub/Sub          Pub/Sub             Pub/Sub         APIs
```

### Technology Stack
- **Backend**: Python 3.8+
- **Message Broker**: Redis 6.0+
- **Data Processing**: NumPy, SciPy, Scikit-learn
- **Visualization**: Plotly, Streamlit
- **Communication**: Redis Pub/Sub, HTTP APIs
- **Data Storage**: Redis Streams, Redis Hash

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Redis Server 6.0 or higher
- Git (for cloning repository)

### System Requirements
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **CPU**: Dual-core processor (Quad-core recommended)
- **Storage**: 2GB free space
- **Network**: Stable internet connection for external APIs

### Installation Steps

#### 1. Install Redis
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install redis-server

# macOS (using Homebrew)
brew install redis

# Windows (using WSL or Redis for Windows)
# Download from: https://redis.io/download
```

#### 2. Start Redis Server
```bash
redis-server
```

#### 3. Install Python Dependencies
```bash
pip install redis pandas numpy scikit-learn plotly streamlit textblob scipy
```

#### 4. Clone and Setup Project
```bash
git clone <repository-url>
cd flood-rescue-system
python -m pip install -r requirements.txt
```

#### 5. Verify Installation
```bash
# Test Redis connection
redis-cli ping
# Should return: PONG

# Test Python imports
python -c "import redis, pandas, numpy, sklearn; print('All dependencies installed')"
```

### Quick Start
```bash
# Terminal 1: Start the multi-agent system
python multi_agent_system.py

# Terminal 2: Start sensor analysis interface
streamlit run sensor_analysis_app.py

# Terminal 3: Start flood simulation (if available)
python flood_simulation.py
```

---

## Agent Documentation

### BaseAgent Class

#### Overview
The `BaseAgent` class provides the foundation for all agents in the system, handling communication, message routing, and lifecycle management.

#### Constructor Parameters
```python
BaseAgent(agent_id: str, agent_type: AgentType, redis_host='localhost', redis_port=6379)
```

- **agent_id**: Unique identifier for the agent
- **agent_type**: Enum value specifying agent type
- **redis_host**: Redis server hostname
- **redis_port**: Redis server port

#### Key Methods

##### `start()`
Starts the agent and begins message processing.
```python
agent.start()
```

##### `stop()`
Gracefully stops the agent.
```python
agent.stop()
```

##### `send_message(recipient_id, message_type, data)`
Sends a message to another agent.
```python
agent.send_message("coordination_agent", "flood_alert", alert_data)
```

##### `process()`
Override this method to implement agent-specific logic.
```python
def process(self):
    # Custom agent logic here
    pass
```

### Sensor Analysis Agent

#### Purpose
Processes IoT sensor data to detect flood conditions and generate spatial flood zones.

#### Key Features
- **Real-time sensor monitoring** with configurable thresholds
- **Spatial clustering** using DBSCAN algorithm
- **Trend analysis** and flood development prediction
- **Network health monitoring** for sensor infrastructure
- **Intelligent alert generation** with cooldown management

#### Configuration Parameters
```python
water_depth_thresholds = {
    'normal': 0.3,      # Below 30cm
    'caution': 0.6,     # 30-60cm  
    'warning': 1.0,     # 60cm-1m
    'critical': 1.5,    # Above 1.5m
    'emergency': 2.0    # Above 2m
}

alert_settings = {
    'min_severity_threshold': 0.4,  # Minimum severity to send alert
    'cooldown_minutes': 15,          # Minutes between alerts for same area
    'zone_expansion_factor': 1.2     # Factor for zone expansion detection
}
```

#### Methods

##### `analyze_sensor_network(sensors)`
Performs comprehensive analysis of sensor network data.
```python
analysis = agent.analyze_sensor_network(sensor_data)
```

**Returns:**
- Network statistics and health metrics
- Detected flood zones with predictions
- Risk assessment and recommendations
- Alert generation summary

##### `detect_flood_zones(sensors, eps=0.01, min_samples=2)`
Detects flood zones using spatial clustering.
```python
zones = agent.detect_flood_zones(sensors, eps=0.01, min_samples=3)
```

**Parameters:**
- **sensors**: List of sensor data dictionaries
- **eps**: DBSCAN clustering distance threshold
- **min_samples**: Minimum sensors required per zone

**Returns:** List of flood zone objects with:
- Geographic boundaries and center coordinates
- Severity metrics and trend analysis
- Confidence scores and predictions
- Associated sensor details

##### `send_zone_alerts(zones)`
Sends flood alerts to coordination agent for significant zones.
```python
agent.send_zone_alerts(detected_zones)
```

#### Alert Generation Logic
1. **Severity Check**: Zone severity must exceed threshold
2. **Cooldown Check**: Must not be in cooldown period for area
3. **Status Check**: Critical/emergency zones always generate alerts
4. **Trend Check**: Worsening zones above moderate severity generate alerts

#### Data Flow
```
Redis Sensor Stream → Fetch Data → Classify Status → Detect Zones → Generate Alerts → Send to Coordination
```

### Tweet Analysis Agent

#### Purpose
Analyzes social media content to detect flood-related discussions and extract location information.

#### Key Features
- **Sentiment analysis** using TextBlob
- **Keyword classification** for flood severity
- **Geographic clustering** of flood-related tweets
- **Confidence scoring** for tweet authenticity
- **Real-time processing** of social media streams

#### Configuration Parameters
```python
flood_keywords = {
    'severe': ['trapped', 'help', 'emergency', 'rescue', 'urgent'],
    'moderate': ['flooding', 'flooded', 'water rising', 'road closed'],
    'mild': ['wet roads', 'puddles', 'rain', 'soggy'],
    'emergency': ['911', 'help', 'sos', 'trapped', 'rescue']
}
```

#### Methods

##### `classify_tweet_severity(text)`
Classifies tweet severity based on content analysis.
```python
severity = agent.classify_tweet_severity("Help! Trapped by flood water!")
# Returns: 0.85 (high severity)
```

##### `detect_flood_clusters(tweets, eps=0.01, min_samples=3)`
Groups nearby flood-related tweets into clusters.
```python
clusters = agent.detect_flood_clusters(tweets)
```

### Coordination Agent

#### Purpose
Receives alerts from analysis agents, makes intelligent decisions about resource allocation, and coordinates emergency response.

#### Key Features
- **Alert clustering** to avoid duplicate responses
- **Priority-based decision making** 
- **Resource tracking** and allocation
- **Multi-source data fusion**
- **Automated coordination decisions**

#### Resource Management
```python
resource_status = {
    'rescue_teams': 5,
    'emergency_vehicles': 10,
    'evacuation_buses': 3,
    'medical_units': 4
}
```

#### Decision Making Process
1. **Alert Reception**: Receives alerts from analysis agents
2. **Spatial Clustering**: Groups nearby alerts to avoid duplication
3. **Priority Assessment**: Determines response priority (1-5 scale)
4. **Resource Allocation**: Assigns appropriate resources
5. **Action Generation**: Creates specific response recommendations
6. **Decision Distribution**: Sends decisions to communication agent

#### Methods

##### `_cluster_alerts()`
Groups nearby alerts into response clusters.
```python
clusters = coordination_agent._cluster_alerts()
```

##### `_make_coordination_decision(alert_cluster)`
Makes response decision for alert cluster.
```python
decision = coordination_agent._make_coordination_decision(cluster)
```

**Decision Factors:**
- Maximum severity in cluster
- Average confidence of alerts
- Geographic spread of alerts
- Available resources
- Historical patterns

#### Message Handlers
- **flood_alert**: Processes incoming flood alerts
- **resource_update**: Updates available resource counts
- **alert_resolved**: Handles alert resolution notifications

### Communication Agent

#### Purpose
Manages all external communications, delivering messages to emergency services, public notifications, and other systems.

#### Key Features
- **Priority-based message queuing**
- **Multi-channel delivery** (radio, phone, email, social media)
- **Rate limiting** to prevent system overload
- **Message persistence** and retry logic
- **Audience-specific messaging**

#### Communication Channels
```python
delivery_channels = {
    'emergency_services': ['radio', 'phone', 'email'],
    'public': ['social_media', 'emergency_broadcast', 'mobile_alert'],
    'agents': ['redis_pubsub']
}
```

#### Message Priority System
- **Priority 1**: Critical emergencies (immediate delivery)
- **Priority 2**: High priority alerts (< 5 minute delay)
- **Priority 3**: Medium priority warnings (< 15 minute delay)
- **Priority 4**: Low priority updates (< 1 hour delay)
- **Priority 5**: Informational messages (best effort)

#### Methods

##### `_generate_messages_from_decision(decision)`
Creates appropriate messages for different audiences.
```python
messages = comm_agent._generate_messages_from_decision(coordination_decision)
```

##### `_deliver_message(message)`
Delivers message through configured channels.
```python
comm_agent._deliver_message(emergency_message)
```

#### Message Processing Flow
```
Coordination Decision → Message Generation → Priority Queuing → Channel Delivery → External Systems
```

---

## Data Structures

### FloodAlert
```python
@dataclass
class FloodAlert:
    id: str                    # Unique alert identifier
    source: str               # 'sensor' or 'tweet'
    location: Dict[str, float] # {'lat': x, 'lon': y}
    severity: float           # 0.0 to 1.0 severity score
    alert_level: AlertLevel   # LOW, MEDIUM, HIGH, CRITICAL
    timestamp: str            # ISO format timestamp
    details: Dict[str, Any]   # Source-specific details
    confidence: float         # 0.0 to 1.0 confidence score
    area_radius: float        # Affected radius in km
```

### CoordinationDecision
```python
@dataclass
class CoordinationDecision:
    decision_id: str                    # Unique decision identifier
    alert_ids: List[str]               # Related alert IDs
    priority: int                      # 1 (highest) to 5 (lowest)
    recommended_actions: List[str]     # Specific action recommendations
    resource_requirements: Dict[str, int]  # Required resources
    affected_area: Dict[str, Any]      # Geographic area info
    timestamp: str                     # Decision timestamp
```

### CommunicationMessage
```python
@dataclass
class CommunicationMessage:
    message_id: str          # Unique message identifier
    recipient_type: str      # 'emergency_services', 'public', 'agents'
    priority: int           # Message priority (1-5)
    content: str            # Message content
    alert_ids: List[str]    # Related alert IDs
    timestamp: str          # Message timestamp
    delivery_channels: List[str]  # Delivery channel list
```

### Sensor Data Format
```python
sensor_data = {
    'sensor_id': str,         # Unique sensor identifier
    'lat': float,            # Latitude coordinate
    'lon': float,            # Longitude coordinate
    'current_reading': float, # Current sensor reading
    'water_depth': float,    # Water depth in meters
    'alert_level': str,      # 'normal', 'caution', 'warning', 'critical', 'emergency'
    'is_flooded': bool,      # Boolean flood status
    'timestamp': str,        # ISO format timestamp
    'battery_level': float,  # Battery percentage (0-100)
    'signal_strength': float # Signal strength (0-100)
}
```

### Tweet Data Format
```python
tweet_data = {
    'id': str,              # Tweet/message ID
    'user_id': str,         # User identifier
    'username': str,        # Username
    'text': str,            # Tweet content
    'lat': float,           # Latitude coordinate
    'lon': float,           # Longitude coordinate
    'timestamp': str,       # ISO format timestamp
    'is_genuine': bool,     # Authenticity flag
    'flood_severity': float, # Calculated severity (0-1)
    'calculated_severity': float # Secondary severity score
}
```

---

## Message Protocols

### Agent Communication Protocol

#### Message Structure
All inter-agent messages follow this structure:
```json
{
    "from": "sender_agent_id",
    "type": "message_type",
    "timestamp": "2025-01-01T12:00:00Z",
    "data": {
        // Message-specific payload
    }
}
```

#### Message Types

##### flood_alert
Sent from analysis agents to coordination agent.
```json
{
    "type": "flood_alert",
    "data": {
        "id": "alert_uuid",
        "source": "sensor",
        "location": {"lat": 32.7767, "lon": -96.7970},
        "severity": 0.8,
        "alert_level": "HIGH",
        "timestamp": "2025-01-01T12:00:00Z",
        "details": {...},
        "confidence": 0.9,
        "area_radius": 1.0
    }
}
```

##### coordination_decision
Sent from coordination agent to communication agent.
```json
{
    "type": "coordination_decision",
    "data": {
        "decision_id": "decision_uuid",
        "alert_ids": ["alert1", "alert2"],
        "priority": 1,
        "recommended_actions": ["Deploy rescue teams", "Evacuate area"],
        "resource_requirements": {"rescue_teams": 2},
        "affected_area": {...},
        "timestamp": "2025-01-01T12:00:00Z"
    }
}
```

##### resource_update
Sent to coordination agent for resource status updates.
```json
{
    "type": "resource_update",
    "data": {
        "rescue_teams": 3,
        "emergency_vehicles": 8,
        "medical_units": 2
    }
}
```

##### alert_resolved
Sent to coordination agent when alerts are resolved.
```json
{
    "type": "alert_resolved",
    "data": {
        "alert_id": "alert_uuid",
        "resolution_time": "2025-01-01T12:30:00Z",
        "resolved_by": "field_team_1"
    }
}
```

### Redis Channel Structure

#### Agent Communication Channels
- `agent:{agent_id}:inbox` - Individual agent inbox
- `agents:broadcast` - Broadcast to all agents

#### External Communication Channels
- `channel:radio` - Emergency services radio
- `channel:phone` - Phone/SMS notifications
- `channel:email` - Email notifications
- `channel:social_media` - Social media posts
- `channel:mobile_alert` - Mobile push notifications

#### Data Storage Keys
- `messages:{recipient_type}` - Message storage hash
- `flood_tweets` - Tweet data stream
- `sensor_data` - Sensor data stream

---

## API Reference

### BaseAgent API

#### Constructor
```python
BaseAgent(agent_id: str, agent_type: AgentType, redis_host='localhost', redis_port=6379)
```

#### Methods

##### start()
```python
def start() -> None:
    """Start the agent and begin processing messages."""
```

##### stop()
```python
def stop() -> None:
    """Stop the agent gracefully."""
```

##### send_message()
```python
def send_message(recipient_id: str, message_type: str, data: Dict[str, Any]) -> None:
    """
    Send message to another agent.
    
    Args:
        recipient_id: Target agent ID or 'broadcast'
        message_type: Type of message being sent
        data: Message payload data
    """
```

##### process()
```python
def process() -> None:
    """
    Main processing logic. Override in subclasses.
    Called continuously while agent is running.
    """
```

### SensorFloodAnalyzer API

#### analyze_sensor_network()
```python
def analyze_sensor_network(sensors: List[Dict]) -> Dict:
    """
    Perform comprehensive sensor network analysis.
    
    Args:
        sensors: List of sensor data dictionaries
        
    Returns:
        Dict containing:
        - network_stats: Overall network statistics
        - network_health: Health and coverage metrics
        - flood_zones: Detected flood zones
        - network_risk: Risk assessment
        - predictions: Forecasting data
        - recommendations: Action recommendations
        - alerts_sent: Alert generation summary
    """
```

#### detect_flood_zones()
```python
def detect_flood_zones(sensors: List[Dict], eps: float = 0.01, min_samples: int = 2) -> List[Dict]:
    """
    Detect flood zones using spatial clustering.
    
    Args:
        sensors: List of sensor data
        eps: DBSCAN clustering distance threshold
        min_samples: Minimum sensors per zone
        
    Returns:
        List of flood zone dictionaries with:
        - Geographic boundaries and center coordinates
        - Severity metrics and confidence scores
        - Trend analysis and predictions
        - Associated sensor details
    """
```

#### classify_sensor_status()
```python
def classify_sensor_status(sensor_data: Dict) -> Tuple[str, float]:
    """
    Classify sensor status based on readings.
    
    Args:
        sensor_data: Sensor data dictionary
        
    Returns:
        Tuple of (status_string, severity_score)
        Status: 'normal', 'caution', 'warning', 'critical', 'emergency'
        Severity: 0.0 to 1.0 float
    """
```

### CoordinationAgent API

#### Resource Management
```python
def update_resources(updates: Dict[str, int]) -> None:
    """Update available resource counts."""

def get_resource_status() -> Dict[str, int]:
    """Get current resource availability."""
```

#### Alert Management
```python
def get_active_alerts() -> Dict[str, FloodAlert]:
    """Get currently active alerts."""

def resolve_alert(alert_id: str) -> None:
    """Mark alert as resolved."""
```

### CommunicationAgent API

#### Message Management
```python
def queue_message(message: CommunicationMessage) -> None:
    """Add message to priority queue."""

def get_queue_status() -> Dict[int, int]:
    """Get message counts by priority."""
```

---

## Configuration

### Environment Variables
```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Agent Configuration
SENSOR_AGENT_ID=sensor_analysis_agent
COORDINATION_AGENT_ID=coordination_agent
COMMUNICATION_AGENT_ID=communication_agent

# Alert Thresholds
MIN_SEVERITY_THRESHOLD=0.4
ALERT_COOLDOWN_MINUTES=15
MAX_ACTIVE_ALERTS=100

# Network Configuration
MAX_SENSORS_PER_ANALYSIS=500
ANALYSIS_TIMEFRAME_HOURS=24
CLUSTER_SENSITIVITY=0.01
MIN_CLUSTER_SIZE=3
```

### Configuration Files

#### config.json
```json
{
    "redis": {
        "host": "localhost",
        "port": 6379,
        "db": 0
    },
    "agents": {
        "sensor_analysis": {
            "agent_id": "sensor_analysis_agent",
            "alert_settings": {
                "min_severity_threshold": 0.4,
                "cooldown_minutes": 15,
                "zone_expansion_factor": 1.2
            },
            "thresholds": {
                "water_depth": {
                    "normal": 0.3,
                    "caution": 0.6,
                    "warning": 1.0,
                    "critical": 1.5,
                    "emergency": 2.0
                }
            }
        },
        "coordination": {
            "agent_id": "coordination_agent",
            "resources": {
                "rescue_teams": 5,
                "emergency_vehicles": 10,
                "evacuation_buses": 3,
                "medical_units": 4
            },
            "decision_timeouts": {
                "alert_expiry_hours": 2,
                "decision_cache_hours": 24
            }
        },
        "communication": {
            "agent_id": "communication_agent",
            "rate_limits": {
                "critical_delay_seconds": 0.1,
                "normal_delay_seconds": 0.5
            },
            "channels": {
                "emergency_services": ["radio", "phone", "email"],
                "public": ["social_media", "mobile_alert"]
            }
        }
    }
}
```

### Loading Configuration
```python
import json

def load_config(config_file='config.json'):
    """Load system configuration from file."""
    with open(config_file, 'r') as f:
        return json.load(f)

# Usage
config = load_config()
agent = SensorFloodAnalyzer(
    alert_settings=config['agents']['sensor_analysis']['alert_settings']
)
```

---

## Deployment Guide

### Production Deployment

#### System Requirements
- **CPU**: 4+ cores recommended
- **Memory**: 8GB+ RAM
- **Storage**: 10GB+ available space
- **Network**: Reliable internet connection
- **OS**: Linux (Ubuntu 20.04+ recommended)

#### Redis Setup
```bash
# Install Redis
sudo apt update
sudo apt install redis-server

# Configure Redis for production
sudo nano /etc/redis/redis.conf

# Key settings:
# maxmemory 4gb
# maxmemory-policy allkeys-lru
# save 900 1
# save 300 10
# save 60 10000

# Start Redis service
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### Python Environment
```bash
# Create virtual environment
python3 -m venv flood_rescue_env
source flood_rescue_env/bin/activate

# Install production dependencies
pip install -r requirements.txt

# Install additional production packages
pip install gunicorn supervisor redis-py-cluster
```

#### Process Management with Supervisor
```bash
# Install supervisor
sudo apt install supervisor

# Create supervisor config
sudo nano /etc/supervisor/conf.d/flood_rescue.conf
```

#### Supervisor Configuration
```ini
[program:flood_coordination_agent]
command=/path/to/flood_rescue_env/bin/python /path/to/coordination_agent.py
directory=/path/to/flood_rescue_system
user=flood_user
autostart=true
autorestart=true
stderr_logfile=/var/log/flood_rescue/coordination_agent.err.log
stdout_logfile=/var/log/flood_rescue/coordination_agent.out.log

[program:flood_communication_agent]
command=/path/to/flood_rescue_env/bin/python /path/to/communication_agent.py
directory=/path/to/flood_rescue_system
user=flood_user
autostart=true
autorestart=true
stderr_logfile=/var/log/flood_rescue/communication_agent.err.log
stdout_logfile=/var/log/flood_rescue/communication_agent.out.log

[program:flood_sensor_agent]
command=/path/to/flood_rescue_env/bin/python /path/to/sensor_agent.py
directory=/path/to/flood_rescue_system
user=flood_user
autostart=true
autorestart=true
stderr_logfile=/var/log/flood_rescue/sensor_agent.err.log
stdout_logfile=/var/log/flood_rescue/sensor_agent.out.log
```

#### Start Services
```bash
# Update supervisor
sudo supervisorctl reread
sudo supervisorctl update

# Start all agents
sudo supervisorctl start all

# Check status
sudo supervisorctl status
```

### Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "multi_agent_system.py"]
```

#### docker-compose.yml
```yaml
version: '3.8'

services:
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  coordination_agent:
    build: .
    depends_on:
      - redis
    environment:
      - REDIS_HOST=redis
      - AGENT_TYPE=coordination
    command: python coordination_agent.py

  communication_agent:
    build: .
    depends_on:
      - redis
    environment:
      - REDIS_HOST=redis
      - AGENT_TYPE=communication
    command: python communication_agent.py

  sensor_agent:
    build: .
    depends_on:
      - redis
    environment:
      - REDIS_HOST=redis
      - AGENT_TYPE=sensor
    command: python sensor_agent.py

  streamlit_app:
    build: .
    depends_on:
      - redis
    ports:
      - "8501:8501"
    environment:
      - REDIS_HOST=redis
    command: streamlit run sensor_analysis_app.py --server.port=8501 --server.address=0.0.0.0

volumes:
  redis_data:
```

#### Deploy with Docker
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale agents
docker-compose up -d --scale sensor_agent=3

# Stop services
docker-compose down
```

### Kubernetes Deployment

#### deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coordination-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: coordination-agent
  template:
    metadata:
      labels:
        app: coordination-agent
    spec:
      containers:
      - name: coordination-agent
        image: flood-rescue:latest
        command: ["python", "coordination_agent.py"]
        env:
        - name: REDIS_HOST
          value: "redis-service"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:6-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc
```

---

## Troubleshooting

### Common Issues

#### Redis Connection Issues
**Problem**: Agents cannot connect to Redis
```
❌ Redis connection failed: Connection refused
```

**Solutions**:
1. Verify Redis is running: `redis-cli ping`
2. Check Redis configuration: `sudo nano /etc/redis/redis.conf`
3. Verify port is open: `netstat -an | grep 6379`
4. Check firewall settings: `sudo ufw status`

#### Agent Communication Issues
**Problem**: Agents not receiving messages
```
Warning: Unknown message type flood_alert in coordination_agent
```

**Solutions**:
1. Verify all agents are started with correct IDs
2. Check Redis pub/sub channels: `redis-cli monitor`
3. Verify message format matches expected structure
4. Check agent logs for initialization errors

#### Memory Issues
**Problem**: System running out of memory
```
MemoryError: Unable to allocate array
```

**Solutions**:
1. Reduce `maxlen` parameters in deque buffers
2. Implement data pruning for old alerts
3. Increase system memory or add swap
4. Optimize clustering algorithms

#### Performance Issues
**Problem**: Slow response times
```
Analysis taking >30 seconds to complete
```

**Solutions**:
1. Reduce data processing window: Set `analysis_hours` to smaller value
2. Optimize clustering parameters: Increase `eps` or reduce `min_samples`
3. Implement data sampling for large datasets
4. Use Redis clustering for horizontal scaling
5. Profile code to identify bottlenecks: `python -m cProfile agent.py`

#### Data Quality Issues
**Problem**: Inconsistent or missing sensor data
```
Error: sensor_data missing required fields
```

**Solutions**:
1. Implement data validation before processing
2. Add default values for missing fields
3. Check data source connectivity
4. Implement data quality monitoring

### Debugging Tools

#### Redis Monitoring
```bash
# Monitor all Redis commands
redis-cli monitor

# Check memory usage
redis-cli info memory

# List all keys
redis-cli keys "*"

# Monitor specific stream
redis-cli xinfo stream sensor_data
```

#### Agent Status Monitoring
```python
def check_agent_status():
    """Check status of all agents"""
    r = redis.Redis()
    
    agents = ['coordination_agent', 'communication_agent', 'sensor_analysis_agent']
    for agent_id in agents:
        try:
            # Check if agent is responsive
            r.publish(f"agent:{agent_id}:inbox", 
                     json.dumps({"type": "ping", "timestamp": datetime.now().isoformat()}))
            print(f"✅ {agent_id}: Active")
        except Exception as e:
            print(f"❌ {agent_id}: Error - {e}")
```

#### Log Analysis
```bash
# Check system logs
sudo journalctl -u flood-rescue-* -f

# Monitor supervisor logs
sudo tail -f /var/log/supervisor/supervisord.log

# Check agent-specific logs
tail -f /var/log/flood_rescue/*.log
```

### Error Recovery

#### Automatic Recovery Strategies
```python
class ResilientAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_count = 0
        self.max_errors = 10
        self.recovery_delay = 5
    
    def _main_loop(self):
        while self.running:
            try:
                self.process()
                self.error_count = 0  # Reset on success
                time.sleep(1)
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in {self.agent_id}: {e}")
                
                if self.error_count >= self.max_errors:
                    logger.critical(f"Max errors reached for {self.agent_id}")
                    self.stop()
                    break
                
                time.sleep(self.recovery_delay)
```

#### Manual Recovery Procedures
1. **Agent Restart**: `sudo supervisorctl restart flood_*`
2. **Redis Restart**: `sudo systemctl restart redis-server`
3. **Full System Restart**: `sudo supervisorctl restart all`
4. **Data Recovery**: Restore from Redis persistence files

---

## Performance Guidelines

### Optimization Strategies

#### Memory Optimization
```python
# Use generators for large datasets
def process_sensors_streaming(sensor_stream):
    for sensor_batch in sensor_stream:
        yield analyze_batch(sensor_batch)

# Implement data pruning
def prune_old_data(self, hours_to_keep=24):
    cutoff_time = datetime.now() - timedelta(hours=hours_to_keep)
    
    # Remove old alerts
    expired_alerts = [
        alert_id for alert_id, alert in self.active_alerts.items()
        if datetime.fromisoformat(alert.timestamp) < cutoff_time
    ]
    
    for alert_id in expired_alerts:
        del self.active_alerts[alert_id]
```

#### CPU Optimization
```python
# Use numpy for vectorized operations
import numpy as np

def calculate_distances_vectorized(locations1, locations2):
    """Fast distance calculation using numpy"""
    loc1_array = np.array(locations1)
    loc2_array = np.array(locations2)
    
    # Vectorized distance calculation
    distances = np.sqrt(np.sum((loc1_array - loc2_array)**2, axis=1)) * 111
    return distances

# Parallel processing for independent tasks
from concurrent.futures import ThreadPoolExecutor

def process_zones_parallel(self, zones):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(self.analyze_zone, zone) for zone in zones]
        results = [future.result() for future in futures]
    return results
```

#### Network Optimization
```python
# Batch message sending
def send_batch_messages(self, messages):
    """Send multiple messages in single Redis pipeline"""
    pipe = self.redis_client.pipeline()
    
    for recipient_id, message_type, data in messages:
        message = {
            'from': self.agent_id,
            'type': message_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        if recipient_id == "broadcast":
            pipe.publish(self.broadcast_channel, json.dumps(message))
        else:
            pipe.publish(f"agent:{recipient_id}:inbox", json.dumps(message))
    
    pipe.execute()
```

### Performance Monitoring

#### Metrics Collection
```python
import time
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def time_function(self, func_name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                execution_time = end_time - start_time
                self.metrics[func_name].append(execution_time)
                
                return result
            return wrapper
        return decorator
    
    def get_performance_stats(self):
        stats = {}
        for func_name, times in self.metrics.items():
            stats[func_name] = {
                'avg_time': np.mean(times),
                'max_time': np.max(times),
                'min_time': np.min(times),
                'call_count': len(times)
            }
        return stats

# Usage
monitor = PerformanceMonitor()

class OptimizedSensorAgent(SensorFloodAnalyzer):
    @monitor.time_function('analyze_network')
    def analyze_sensor_network(self, sensors):
        return super().analyze_sensor_network(sensors)
```

#### Benchmarking
```python
def benchmark_system(num_sensors=1000, num_tweets=500):
    """Benchmark system performance"""
    
    # Generate test data
    test_sensors = generate_test_sensors(num_sensors)
    test_tweets = generate_test_tweets(num_tweets)
    
    # Benchmark sensor analysis
    start_time = time.time()
    sensor_agent = SensorFloodAnalyzer()
    sensor_analysis = sensor_agent.analyze_sensor_network(test_sensors)
    sensor_time = time.time() - start_time
    
    # Benchmark coordination
    start_time = time.time()
    coord_agent = CoordinationAgent()
    # Simulate alert processing
    coord_time = time.time() - start_time
    
    return {
        'sensor_analysis_time': sensor_time,
        'coordination_time': coord_time,
        'total_time': sensor_time + coord_time,
        'sensors_per_second': num_sensors / sensor_time
    }
```

### Scalability Guidelines

#### Horizontal Scaling
```python
# Multiple sensor analysis agents
class ScalableSensorSystem:
    def __init__(self, num_agents=3):
        self.agents = []
        for i in range(num_agents):
            agent = SensorFloodAnalyzer(f"sensor_agent_{i}")
            self.agents.append(agent)
    
    def distribute_sensors(self, sensors):
        """Distribute sensors across agents"""
        chunk_size = len(sensors) // len(self.agents)
        
        for i, agent in enumerate(self.agents):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < len(self.agents) - 1 else len(sensors)
            agent_sensors = sensors[start_idx:end_idx]
            
            # Process in separate thread
            threading.Thread(
                target=agent.analyze_sensor_network,
                args=(agent_sensors,)
            ).start()
```

#### Vertical Scaling
```python
# Optimize for multi-core systems
import multiprocessing as mp

def parallel_zone_detection(sensors, num_processes=None):
    """Detect zones using multiple processes"""
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Split sensors into chunks
    chunk_size = len(sensors) // num_processes
    sensor_chunks = [
        sensors[i:i + chunk_size] 
        for i in range(0, len(sensors), chunk_size)
    ]
    
    # Process chunks in parallel
    with mp.Pool(num_processes) as pool:
        zone_results = pool.map(detect_zones_chunk, sensor_chunks)
    
    # Merge results
    all_zones = []
    for zones in zone_results:
        all_zones.extend(zones)
    
    return all_zones
```

---

## Extension Guide

### Adding New Agent Types

#### Step 1: Define Agent Class
```python
class RouteOptimizationAgent(BaseAgent):
    def __init__(self, agent_id="route_optimization_agent"):
        super().__init__(agent_id, AgentType.ROUTE_OPTIMIZATION)
        
        # Agent-specific initialization
        self.road_network = self.load_road_network()
        self.blocked_routes = set()
        
        # Message handlers
        self.message_handlers = {
            'coordination_decision': self._handle_coordination_decision,
            'road_blocked': self._handle_road_blocked,
            'route_request': self._handle_route_request
        }
    
    def _handle_coordination_decision(self, message):
        """Process coordination decisions to plan routes"""
        decision = CoordinationDecision(**message['data'])
        
        # Calculate optimal routes for emergency vehicles
        routes = self.calculate_emergency_routes(decision.affected_area)
        
        # Send route recommendations
        route_data = {
            'decision_id': decision.decision_id,
            'recommended_routes': routes,
            'estimated_times': self.calculate_travel_times(routes)
        }
        
        self.send_message("communication_agent", "route_update", route_data)
```

#### Step 2: Register Agent Type
```python
class AgentType(Enum):
    SENSOR_AGENT = "sensor_agent"
    TWEET_AGENT = "tweet_agent"
    COORDINATION_AGENT = "coordination_agent"
    COMMUNICATION_AGENT = "communication_agent"
    ROUTE_OPTIMIZATION = "route_optimization_agent"  # New agent type
```

#### Step 3: Integrate with System
```python
class ExtendedFloodRescueSystem(FloodRescueSystem):
    def start_system(self):
        super().start_system()
        
        # Add new agent
        self.agents['route_optimization'] = RouteOptimizationAgent()
        self.agents['route_optimization'].start()
```

### Adding New Data Sources

#### Step 1: Create Data Adapter
```python
class WeatherDataAdapter:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.weather.com/v1"
    
    def fetch_weather_data(self, lat, lon):
        """Fetch weather data for location"""
        response = requests.get(
            f"{self.base_url}/current/conditions",
            params={
                'key': self.api_key,
                'q': f"{lat},{lon}",
                'format': 'json'
            }
        )
        return response.json()
    
    def to_flood_risk_score(self, weather_data):
        """Convert weather data to flood risk score"""
        rainfall = weather_data.get('precipitation', 0)
        wind_speed = weather_data.get('wind_speed', 0)
        
        # Simple risk calculation
        risk_score = min((rainfall * 0.1 + wind_speed * 0.02), 1.0)
        return risk_score
```

#### Step 2: Create Specialized Agent
```python
class WeatherAnalysisAgent(BaseAgent):
    def __init__(self, agent_id="weather_analysis_agent", api_key=None):
        super().__init__(agent_id, AgentType.WEATHER_AGENT)
        self.weather_adapter = WeatherDataAdapter(api_key)
        self.monitoring_locations = []
    
    def process(self):
        """Continuously monitor weather conditions"""
        for location in self.monitoring_locations:
            weather_data = self.weather_adapter.fetch_weather_data(
                location['lat'], location['lon']
            )
            
            risk_score = self.weather_adapter.to_flood_risk_score(weather_data)
            
            if risk_score > 0.6:  # High weather-based flood risk
                alert = FloodAlert(
                    id=str(uuid.uuid4()),
                    source='weather',
                    location=location,
                    severity=risk_score,
                    alert_level=AlertLevel.MEDIUM,
                    timestamp=datetime.now().isoformat(),
                    details=weather_data,
                    confidence=0.7,
                    area_radius=5.0  # Larger radius for weather events
                )
                
                self.send_message("coordination_agent", "flood_alert", asdict(alert))
        
        time.sleep(300)  # Check every 5 minutes
```

### Adding New Communication Channels

#### Step 1: Create Channel Handler
```python
class SlackChannelHandler:
    def __init__(self, bot_token, channel_id):
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.client = SlackClient(bot_token)
    
    def send_message(self, message):
        """Send message to Slack channel"""
        try:
            response = self.client.chat_postMessage(
                channel=self.channel_id,
                text=message.content,
                username="Flood Alert Bot",
                icon_emoji=":warning:"
            )
            return response['ok']
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False
```

#### Step 2: Integrate with Communication Agent
```python
class ExtendedCommunicationAgent(CommunicationAgent):
    def __init__(self, agent_id="communication_agent"):
        super().__init__(agent_id)
        
        # Add new channel handlers
        self.channel_handlers = {
            'slack': SlackChannelHandler(bot_token, channel_id),
            'discord': DiscordChannelHandler(bot_token, channel_id),
            'teams': TeamsChannelHandler(webhook_url)
        }
        
        # Update delivery channels
        self.delivery_channels['emergency_services'].append('slack')
        self.delivery_channels['public'].append('discord')
    
    def _deliver_message(self, message):
        """Extended message delivery with new channels"""
        super()._deliver_message(message)
        
        # Handle new channels
        for channel in message.delivery_channels:
            if channel in self.channel_handlers:
                handler = self.channel_handlers[channel]
                success = handler.send_message(message)
                
                if success:
                    logger.info(f"Message delivered via {channel}")
                else:
                    logger.error(f"Failed to deliver message via {channel}")
```

### Custom Alert Processing

#### Step 1: Create Alert Processor
```python
class CustomAlertProcessor:
    def __init__(self):
        self.processing_rules = []
    
    def add_rule(self, condition_func, action_func):
        """Add custom processing rule"""
        self.processing_rules.append({
            'condition': condition_func,
            'action': action_func
        })
    
    def process_alert(self, alert):
        """Process alert through custom rules"""
        for rule in self.processing_rules:
            if rule['condition'](alert):
                modified_alert = rule['action'](alert)
                if modified_alert:
                    return modified_alert
        
        return alert

# Example usage
processor = CustomAlertProcessor()

# Rule: Escalate alerts near schools
def near_school_condition(alert):
    # Check if alert is near a school
    return is_near_school(alert.location)

def escalate_school_alert(alert):
    # Increase severity for school areas
    alert.severity = min(alert.severity * 1.5, 1.0)
    alert.alert_level = AlertLevel.HIGH
    alert.details['reason'] = 'Near school area'
    return alert

processor.add_rule(near_school_condition, escalate_school_alert)
```

#### Step 2: Integrate with Agents
```python
class CustomCoordinationAgent(CoordinationAgent):
    def __init__(self, agent_id="coordination_agent"):
        super().__init__(agent_id)
        self.alert_processor = CustomAlertProcessor()
        self.setup_custom_rules()
    
    def setup_custom_rules(self):
        """Setup custom alert processing rules"""
        # Add various custom rules
        self.alert_processor.add_rule(
            lambda alert: alert.severity > 0.9,
            self.handle_extreme_severity
        )
    
    def _handle_flood_alert(self, message):
        """Handle alerts with custom processing"""
        alert_data = message['data']
        alert = FloodAlert(**alert_data)
        
        # Apply custom processing
        processed_alert = self.alert_processor.process_alert(alert)
        
        self.active_alerts[processed_alert.id] = processed_alert
        logger.info(f"Received and processed flood alert {processed_alert.id}")
        
        self._analyze_and_coordinate()
```

### Machine Learning Integration

#### Step 1: Create ML Models
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

class FloodPredictionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, sensor_data, weather_data=None):
        """Prepare features for ML model"""
        features = []
        
        for sensor in sensor_data:
            feature_vector = [
                sensor['water_depth'],
                sensor['current_reading'],
                sensor['lat'],
                sensor['lon'],
                # Add more features as needed
            ]
            
            if weather_data:
                feature_vector.extend([
                    weather_data.get('rainfall', 0),
                    weather_data.get('wind_speed', 0),
                    weather_data.get('pressure', 1013.25)
                ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def train(self, historical_data, labels):
        """Train the flood prediction model"""
        features = self.prepare_features(historical_data)
        features_scaled = self.scaler.fit_transform(features)
        
        self.model.fit(features_scaled, labels)
        self.is_trained = True
    
    def predict_flood_probability(self, sensor_data):
        """Predict flood probability for current conditions"""
        if not self.is_trained:
            return 0.5  # Default uncertainty
        
        features = self.prepare_features(sensor_data)
        features_scaled = self.scaler.transform(features)
        
        probabilities = self.model.predict_proba(features_scaled)
        return probabilities[:, 1]  # Probability of flood class
    
    def save_model(self, filepath):
        """Save trained model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
```

#### Step 2: Integrate ML with Agents
```python
class MLEnhancedSensorAgent(SensorFloodAnalyzer):
    def __init__(self, agent_id="ml_sensor_agent"):
        super().__init__(agent_id)
        self.prediction_model = FloodPredictionModel()
        
        # Load pre-trained model if available
        try:
            self.prediction_model.load_model('flood_model.pkl')
        except FileNotFoundError:
            logger.warning("No pre-trained model found. Using default analysis.")
    
    def analyze_sensor_network(self, sensors):
        """Enhanced analysis with ML predictions"""
        # Standard analysis
        standard_analysis = super().analyze_sensor_network(sensors)
        
        # ML-enhanced predictions
        if self.prediction_model.is_trained:
            flood_probabilities = self.prediction_model.predict_flood_probability(sensors)
            
            # Adjust severity scores based on ML predictions
            for i, sensor in enumerate(sensors):
                ml_probability = flood_probabilities[i]
                
                # Combine traditional analysis with ML prediction
                combined_severity = (sensor['severity_score'] + ml_probability) / 2
                sensor['ml_enhanced_severity'] = combined_severity
                
                # Update alert generation based on ML insights
                if ml_probability > 0.8 and sensor['severity_score'] < 0.4:
                    # ML detected high risk that traditional analysis missed
                    self.send_ml_enhanced_alert(sensor, ml_probability)
        
        # Add ML metrics to analysis
        standard_analysis['ml_insights'] = {
            'model_available': self.prediction_model.is_trained,
            'predictions_made': len(sensors) if self.prediction_model.is_trained else 0,
            'high_risk_predictions': len([p for p in flood_probabilities if p > 0.7]) if self.prediction_model.is_trained else 0
        }
        
        return standard_analysis
    
    def send_ml_enhanced_alert(self, sensor, ml_probability):
        """Send alert based on ML prediction"""
        alert = FloodAlert(
            id=str(uuid.uuid4()),
            source='sensor_ml',
            location={'lat': sensor['lat'], 'lon': sensor['lon']},
            severity=ml_probability,
            alert_level=AlertLevel.HIGH,
            timestamp=datetime.now().isoformat(),
            details={
                'sensor_id': sensor['sensor_id'],
                'ml_probability': ml_probability,
                'traditional_severity': sensor['severity_score'],
                'model_type': 'RandomForest'
            },
            confidence=0.85,
            area_radius=0.5
        )
        
        self.send_message("coordination_agent", "flood_alert", asdict(alert))
```

---

## Conclusion

This documentation provides comprehensive guidance for deploying, configuring, and extending the Multi-Agent Flood Rescue System. The system's modular architecture allows for easy customization and scaling to meet specific deployment requirements.

For additional support or to contribute to the project, please refer to the project repository or contact the development team.

### Quick Reference Links
- [System Architecture](#architecture)
- [API Reference](#api-reference)  
- [Configuration Guide](#configuration)
- [Deployment Instructions](#deployment-guide)
- [Troubleshooting](#troubleshooting)
- [Extension Examples](#extension-guide)

### Version Information
- **Documentation Version**: 1.0.0
- **System Version**: 1.0.0
- **Last Updated**: January 2025
- **Python Compatibility**: 3.8+
- **Redis Compatibility**: 6.0+