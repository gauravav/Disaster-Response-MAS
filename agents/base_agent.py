class BaseAgent:
    def __init__(self, unique_id):
        self.unique_id = unique_id

    def step(self):
        raise NotImplementedError("Each agent must implement the step method.")
