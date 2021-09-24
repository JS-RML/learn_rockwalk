import pybullet as bullet
import os
import numpy as np


class GoalMarker:
    def __init__(self, clientID):
        self.clientID = clientID

    def load_model_from_urdf(self):
        f_name = os.path.join(os.path.dirname(__file__), 'models/goal_direction.urdf')

        self.goalID = bullet.loadURDF(fileName=f_name, basePosition=[7.5, 0.0, -0.04], physicsClientId=self.clientID)
