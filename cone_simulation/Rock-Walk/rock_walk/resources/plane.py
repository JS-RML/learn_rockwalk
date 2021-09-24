import pybullet as bullet
import os


class Plane:
    def __init__(self, clientID):
        self.clientID = clientID

    def load_model_from_urdf(self):
        f_name1 = os.path.join(os.path.dirname(__file__), 'models/plane.urdf')

        self.planeID = bullet.loadURDF(fileName=f_name1, basePosition=[0, 0, 0], physicsClientId=self.clientID)

    def load_texutre(self):
        f_name2 = os.path.join(os.path.dirname(__file__), 'models/texture/checker_blue.png')
        texID = bullet.loadTexture(f_name2)
        bullet.changeVisualShape(self.planeID, -1, textureUniqueId=texID, physicsClientId=self.clientID)

    def get_ids(self):
        return self.planeID, self.clientID


    def get_dynamics_info(self):
        print(bullet.getDynamicsInfo(self.planeID, -1, self.clientID))

    def set_lateral_friction(self, value):
        bullet.changeDynamics(self.planeID, -1, lateralFriction=value, physicsClientId=self.clientID)
