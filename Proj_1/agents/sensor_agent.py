from Proj_1.agents.base_agent import BaseAgent
from Proj_1.simulation.event_queue import event_queue_v3

class SensorAgent(BaseAgent):
    def __init__(self, unique_id, sensor_data, sensor_threshold=2.0):
        super().__init__(unique_id)
        self.sensor_data = sensor_data
        self.sensor_threshold = sensor_threshold

    def step(self):
        for _, row in self.sensor_data.iterrows():
            if row['value'] > self.sensor_threshold:
                print("Spike Threshold:", self.sensor_threshold)
                print("Sensor spike detected at ({}, {}): {}".format(row['x'], row['y'], row['value']))
                event_queue_v3.append({
                    "type": "sensor",
                    "location": (int(row['x']), int(row['y'])),
                    "value": row['value'],
                    "agent": self.unique_id
                })
