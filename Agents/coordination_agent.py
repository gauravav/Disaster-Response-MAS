import redis
import json
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from enum import Enum
import logging
from collections import defaultdict, deque
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    location: Dict[str, float]  # {'lat': x, 'lon': y}
    severity: float  # 0.0 to 1.0
    alert_level: AlertLevel
    timestamp: str
    details: Dict[str, Any]
    confidence: float
    area_radius: float = 0.5  # km

@dataclass
class CoordinationDecision:
    decision_id: str
    alert_ids: List[str]
    priority: int  # 1 (highest) to 5 (lowest)
    recommended_actions: List[str]
    resource_requirements: Dict[str, int]
    affected_area: Dict[str, Any]
    timestamp: str

@dataclass
class CommunicationMessage:
    message_id: str
    recipient_type: str  # 'emergency_services', 'public', 'agents'
    priority: int
    content: str
    alert_ids: List[str]
    timestamp: str
    delivery_channels: List[str]

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

    def start(self):
        """Start the agent"""
        self.running = True
        logger.info(f"Agent {self.agent_id} ({self.agent_type.value}) starting...")

        # Start message listener in separate thread
        self.message_thread = threading.Thread(target=self._listen_for_messages)
        self.message_thread.daemon = True
        self.message_thread.start()

        # Start main agent loop
        self.main_thread = threading.Thread(target=self._main_loop)
        self.main_thread.daemon = True
        self.main_thread.start()

    def stop(self):
        """Stop the agent"""
        self.running = False
        logger.info(f"Agent {self.agent_id} stopping...")

    def send_message(self, recipient_id: str, message_type: str, data: Dict[str, Any]):
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

    def _listen_for_messages(self):
        """Listen for incoming messages"""
        while self.running:
            try:
                message = self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    self._handle_message(json.loads(message['data']))
            except Exception as e:
                logger.error(f"Error handling message in {self.agent_id}: {e}")

    def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming message"""
        message_type = message.get('type')
        if message_type in self.message_handlers:
            self.message_handlers[message_type](message)
        else:
            logger.warning(f"Unknown message type {message_type} in {self.agent_id}")

    def _main_loop(self):
        """Main agent processing loop - to be overridden"""
        while self.running:
            try:
                self.process()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in {self.agent_id} main loop: {e}")

    def process(self):
        """Main processing logic - to be overridden"""
        pass

class CoordinationAgent(BaseAgent):
    def __init__(self, agent_id="coordination_agent"):
        super().__init__(agent_id, AgentType.COORDINATION_AGENT)

        # Storage for alerts and decisions
        self.active_alerts = {}  # alert_id -> FloodAlert
        self.decisions = deque(maxlen=100)
        self.resource_status = {
            'rescue_teams': 5,
            'emergency_vehicles': 10,
            'evacuation_buses': 3,
            'medical_units': 4
        }

        # Message handlers
        self.message_handlers = {
            'flood_alert': self._handle_flood_alert,
            'resource_update': self._handle_resource_update,
            'alert_resolved': self._handle_alert_resolved
        }

        logger.info("Coordination Agent initialized")

    def _handle_flood_alert(self, message):
        """Handle incoming flood alerts from sensor/tweet agents"""
        alert_data = message['data']
        alert = FloodAlert(**alert_data)

        self.active_alerts[alert.id] = alert
        logger.info(f"Received flood alert {alert.id} from {alert.source}")

        # Trigger coordination analysis
        self._analyze_and_coordinate()

    def _handle_resource_update(self, message):
        """Handle resource status updates"""
        updates = message['data']
        self.resource_status.update(updates)
        logger.info(f"Resource status updated: {updates}")

    def _handle_alert_resolved(self, message):
        """Handle alert resolution"""
        alert_id = message['data']['alert_id']
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} resolved")

    def _analyze_and_coordinate(self):
        """Analyze current alerts and make coordination decisions"""
        if not self.active_alerts:
            return

        # Group nearby alerts
        alert_clusters = self._cluster_alerts()

        for cluster in alert_clusters:
            decision = self._make_coordination_decision(cluster)
            self.decisions.append(decision)

            # Send decision to communication agent
            self.send_message("communication_agent", "coordination_decision", asdict(decision))

            # Log decision
            logger.info(f"Coordination decision {decision.decision_id}: Priority {decision.priority}")

    def _cluster_alerts(self) -> List[List[FloodAlert]]:
        """Group nearby alerts into clusters"""
        alerts = list(self.active_alerts.values())
        clusters = []
        processed = set()

        for alert in alerts:
            if alert.id in processed:
                continue

            cluster = [alert]
            processed.add(alert.id)

            # Find nearby alerts
            for other_alert in alerts:
                if other_alert.id in processed:
                    continue

                distance = self._calculate_distance(alert.location, other_alert.location)
                if distance <= max(alert.area_radius, other_alert.area_radius):
                    cluster.append(other_alert)
                    processed.add(other_alert.id)

            clusters.append(cluster)

        return clusters

    def _calculate_distance(self, loc1: Dict[str, float], loc2: Dict[str, float]) -> float:
        """Calculate distance between two locations (simplified)"""
        lat_diff = loc1['lat'] - loc2['lat']
        lon_diff = loc1['lon'] - loc2['lon']
        return (lat_diff**2 + lon_diff**2)**0.5 * 111  # Rough km conversion

    def _make_coordination_decision(self, alert_cluster: List[FloodAlert]) -> CoordinationDecision:
        """Make coordination decision for alert cluster"""
        # Calculate cluster properties
        max_severity = max(alert.severity for alert in alert_cluster)
        avg_confidence = sum(alert.confidence for alert in alert_cluster) / len(alert_cluster)

        # Determine priority (1 = highest, 5 = lowest)
        if max_severity >= 0.8:
            priority = 1
        elif max_severity >= 0.6:
            priority = 2
        elif max_severity >= 0.4:
            priority = 3
        else:
            priority = 4

        # Generate recommendations
        actions = []
        resources = {}

        if max_severity >= 0.8:
            actions.extend([
                "Deploy immediate rescue teams",
                "Initiate evacuation procedures",
                "Alert emergency services"
            ])
            resources = {'rescue_teams': 2, 'emergency_vehicles': 3, 'medical_units': 1}
        elif max_severity >= 0.6:
            actions.extend([
                "Monitor area closely",
                "Prepare evacuation routes",
                "Deploy monitoring team"
            ])
            resources = {'rescue_teams': 1, 'emergency_vehicles': 1}
        else:
            actions.extend([
                "Continue monitoring",
                "Issue area warning"
            ])
            resources = {'emergency_vehicles': 1}

        # Calculate affected area
        lats = [alert.location['lat'] for alert in alert_cluster]
        lons = [alert.location['lon'] for alert in alert_cluster]

        affected_area = {
            'center_lat': sum(lats) / len(lats),
            'center_lon': sum(lons) / len(lons),
            'radius': max(alert.area_radius for alert in alert_cluster)
        }

        return CoordinationDecision(
            decision_id=str(uuid.uuid4()),
            alert_ids=[alert.id for alert in alert_cluster],
            priority=priority,
            recommended_actions=actions,
            resource_requirements=resources,
            affected_area=affected_area,
            timestamp=datetime.now().isoformat()
        )

    def process(self):
        """Main coordination processing"""
        # Periodic cleanup of old alerts
        current_time = datetime.now()
        expired_alerts = []

        for alert_id, alert in self.active_alerts.items():
            alert_time = datetime.fromisoformat(alert.timestamp)
            if current_time - alert_time > timedelta(hours=2):  # 2 hour timeout
                expired_alerts.append(alert_id)

        for alert_id in expired_alerts:
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} expired")

class CommunicationAgent(BaseAgent):
    def __init__(self, agent_id="communication_agent"):
        super().__init__(agent_id, AgentType.COMMUNICATION_AGENT)

        # Message queues for different priorities
        self.message_queues = {
            1: deque(),  # Critical
            2: deque(),  # High
            3: deque(),  # Medium
            4: deque(),  # Low
            5: deque()   # Informational
        }

        # Communication channels
        self.delivery_channels = {
            'emergency_services': ['radio', 'phone', 'email'],
            'public': ['social_media', 'emergency_broadcast', 'mobile_alert'],
            'agents': ['redis_pubsub']
        }

        # Message handlers
        self.message_handlers = {
            'coordination_decision': self._handle_coordination_decision,
            'emergency_alert': self._handle_emergency_alert,
            'status_update': self._handle_status_update
        }

        logger.info("Communication Agent initialized")

    def _handle_coordination_decision(self, message):
        """Handle coordination decisions and generate communications"""
        decision_data = message['data']
        decision = CoordinationDecision(**decision_data)

        # Generate messages for different recipients
        messages = self._generate_messages_from_decision(decision)

        for msg in messages:
            self.message_queues[msg.priority].append(msg)
            logger.info(f"Queued message {msg.message_id} for {msg.recipient_type}")

    def _handle_emergency_alert(self, message):
        """Handle direct emergency alerts"""
        alert_data = message['data']

        # Create high-priority emergency message
        emergency_msg = CommunicationMessage(
            message_id=str(uuid.uuid4()),
            recipient_type='emergency_services',
            priority=1,
            content=f"EMERGENCY: {alert_data.get('description', 'Flood emergency detected')}",
            alert_ids=[alert_data.get('alert_id', '')],
            timestamp=datetime.now().isoformat(),
            delivery_channels=['radio', 'phone']
        )

        self.message_queues[1].append(emergency_msg)

    def _handle_status_update(self, message):
        """Handle status updates"""
        update_data = message['data']

        status_msg = CommunicationMessage(
            message_id=str(uuid.uuid4()),
            recipient_type='agents',
            priority=4,
            content=f"Status update: {update_data.get('message', '')}",
            alert_ids=[],
            timestamp=datetime.now().isoformat(),
            delivery_channels=['redis_pubsub']
        )

        self.message_queues[4].append(status_msg)

    def _generate_messages_from_decision(self, decision: CoordinationDecision) -> List[CommunicationMessage]:
        """Generate appropriate messages from coordination decision"""
        messages = []

        # Emergency services message
        if decision.priority <= 2:
            emergency_content = f"""
FLOOD ALERT - Priority {decision.priority}
Location: {decision.affected_area['center_lat']:.4f}, {decision.affected_area['center_lon']:.4f}
Actions: {', '.join(decision.recommended_actions)}
Resources needed: {decision.resource_requirements}
Alert IDs: {', '.join(decision.alert_ids)}
            """.strip()

            messages.append(CommunicationMessage(
                message_id=str(uuid.uuid4()),
                recipient_type='emergency_services',
                priority=decision.priority,
                content=emergency_content,
                alert_ids=decision.alert_ids,
                timestamp=datetime.now().isoformat(),
                delivery_channels=['radio', 'email', 'phone']
            ))

        # Public warning message
        if decision.priority <= 3:
            public_content = f"""
FLOOD WARNING: Flooding detected in your area.
Location: Near {decision.affected_area['center_lat']:.2f}, {decision.affected_area['center_lon']:.2f}
Recommendation: {"Evacuate immediately" if decision.priority == 1 else "Avoid the area and stay alert"}
Time: {datetime.now().strftime('%H:%M')}
            """.strip()

            messages.append(CommunicationMessage(
                message_id=str(uuid.uuid4()),
                recipient_type='public',
                priority=decision.priority,
                content=public_content,
                alert_ids=decision.alert_ids,
                timestamp=datetime.now().isoformat(),
                delivery_channels=['mobile_alert', 'social_media']
            ))

        # Agent coordination message
        agent_content = f"""
COORDINATION UPDATE
Decision ID: {decision.decision_id}
Priority: {decision.priority}
Affected Area: {decision.affected_area}
Actions: {decision.recommended_actions}
        """.strip()

        messages.append(CommunicationMessage(
            message_id=str(uuid.uuid4()),
            recipient_type='agents',
            priority=decision.priority + 1,  # Lower priority for agents
            content=agent_content,
            alert_ids=decision.alert_ids,
            timestamp=datetime.now().isoformat(),
            delivery_channels=['redis_pubsub']
        ))

        return messages

    def process(self):
        """Process message queues and deliver messages"""
        # Process messages by priority (1 = highest priority first)
        for priority in sorted(self.message_queues.keys()):
            queue = self.message_queues[priority]

            while queue:
                message = queue.popleft()
                self._deliver_message(message)

                # Rate limiting - don't overwhelm systems
                if priority == 1:
                    time.sleep(0.1)  # Critical messages - slight delay
                else:
                    time.sleep(0.5)  # Other messages - longer delay
                break  # Process one message per cycle to maintain responsiveness

    def _deliver_message(self, message: CommunicationMessage):
        """Deliver message through appropriate channels"""
        logger.info(f"Delivering message {message.message_id} to {message.recipient_type}")

        # Store message in Redis for external systems to consume
        self.redis_client.hset(
            f"messages:{message.recipient_type}",
            message.message_id,
            json.dumps(asdict(message))
        )

        # Set expiration (24 hours)
        self.redis_client.expire(f"messages:{message.recipient_type}", 86400)

        # Publish to specific channels
        for channel in message.delivery_channels:
            channel_key = f"channel:{channel}"
            self.redis_client.publish(channel_key, json.dumps(asdict(message)))

        logger.info(f"Message {message.message_id} delivered via {message.delivery_channels}")

# Example integration with your existing agents
class SensorAnalysisAgent(BaseAgent):
    """Your existing sensor agent - modified to send alerts"""
    def __init__(self, agent_id="sensor_analysis_agent"):
        super().__init__(agent_id, AgentType.SENSOR_AGENT)

    def process(self):
        """Your existing sensor analysis logic"""
        # Your existing sensor analysis code here
        # When you detect a flood, send alert to coordination agent

        # Example flood detection
        flood_detected = self._analyze_sensors()  # Your existing method

        if flood_detected:
            alert = FloodAlert(
                id=str(uuid.uuid4()),
                source='sensor',
                location={'lat': 32.7767, 'lon': -96.7970},  # Example coordinates
                severity=0.8,
                alert_level=AlertLevel.HIGH,
                timestamp=datetime.now().isoformat(),
                details={'sensor_id': 'sensor_001', 'water_level': 2.5},
                confidence=0.9,
                area_radius=1.0
            )

            # Send to coordination agent
            self.send_message("coordination_agent", "flood_alert", asdict(alert))

    def _analyze_sensors(self):
        """Your existing sensor analysis logic"""
        # Replace with your actual sensor analysis
        return False  # Placeholder

class TweetAnalysisAgent(BaseAgent):
    """Your existing tweet agent - modified to send alerts"""
    def __init__(self, agent_id="tweet_analysis_agent"):
        super().__init__(agent_id, AgentType.TWEET_AGENT)

    def process(self):
        """Your existing tweet analysis logic"""
        # Your existing tweet analysis code here
        # When you detect flood tweets, send alert to coordination agent

        flood_cluster = self._analyze_tweets()  # Your existing method

        if flood_cluster:
            alert = FloodAlert(
                id=str(uuid.uuid4()),
                source='tweet',
                location={'lat': flood_cluster['center_lat'], 'lon': flood_cluster['center_lon']},
                severity=flood_cluster['severity'],
                alert_level=AlertLevel.MEDIUM,
                timestamp=datetime.now().isoformat(),
                details={'tweet_count': flood_cluster['tweet_count']},
                confidence=flood_cluster['confidence'],
                area_radius=flood_cluster['radius']
            )

            # Send to coordination agent
            self.send_message("coordination_agent", "flood_alert", asdict(alert))

    def _analyze_tweets(self):
        """Your existing tweet analysis logic"""
        # Replace with your actual tweet analysis
        return None  # Placeholder

# Main system controller
class FloodRescueSystem:
    def __init__(self):
        self.agents = {}

    def start_system(self):
        """Start all agents"""
        logger.info("Starting Flood Rescue Multi-Agent System...")

        # Initialize agents
        self.agents['coordination'] = CoordinationAgent()
        self.agents['communication'] = CommunicationAgent()
        self.agents['sensor_analysis'] = SensorAnalysisAgent()
        self.agents['tweet_analysis'] = TweetAnalysisAgent()

        # Start all agents
        for agent in self.agents.values():
            agent.start()

        logger.info("All agents started successfully!")

    def stop_system(self):
        """Stop all agents"""
        logger.info("Stopping Flood Rescue Multi-Agent System...")

        for agent in self.agents.values():
            agent.stop()

        logger.info("All agents stopped.")

    def get_system_status(self):
        """Get status of all agents"""
        status = {}
        for name, agent in self.agents.items():
            status[name] = {
                'agent_id': agent.agent_id,
                'type': agent.agent_type.value,
                'running': agent.running
            }
        return status

# Usage example
if __name__ == "__main__":
    # Create and start the multi-agent system
    system = FloodRescueSystem()

    try:
        system.start_system()

        # Keep system running
        while True:
            time.sleep(10)
            status = system.get_system_status()
            logger.info(f"System status: {status}")

    except KeyboardInterrupt:
        logger.info("Shutting down system...")
        system.stop_system()