from Proj_1.agents.social_media_agent import SocialMediaAgent
from Proj_1.agents.sensor_agent import SensorAgent

class DisasterModelV3:
    def __init__(self, tweets, sensor_data, sensor_threshold):
        self.agents = []
        self.agents.append(SocialMediaAgent("social-1", tweets))
        self.agents.append(SensorAgent("sensor-1", sensor_data, sensor_threshold))

    def step(self):
        for agent in self.agents:
            agent.step()
