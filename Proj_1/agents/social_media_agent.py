import re
from Proj_1.agents.base_agent import BaseAgent
from Proj_1.simulation.event_queue import event_queue_v3

class SocialMediaAgent(BaseAgent):
    def __init__(self, unique_id, tweet_source):
        super().__init__(unique_id)
        self.tweet_source = tweet_source

    def step(self):
        for tweet in self.tweet_source[:50]:
            tweet_text = tweet['text'] if isinstance(tweet, dict) else tweet
            match = re.search(r'zone (\d+),(\d+)', tweet_text)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                event_queue_v3.append({
                    "type": "social",
                    "location": (x, y),
                    "message": tweet_text,
                    "agent": self.unique_id
                })