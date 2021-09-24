import gym
import math
import time
import numpy as np
import pybullet as bullet

from rock_walk.resources.trainingObject import TrainObject
from rock_walk.resources.plane import Plane
from rock_walk.resources.goal import GoalMarker


EPISODE_TIMEOUT = 5 #seconds
DESIRED_NUTATION = np.radians(20)

class RockWalkEnv(gym.Env):

    def __init__(self, bullet_connection, step_freq, frame_skip, isTrain):
        self._bullet_connection = bullet_connection
        self._frame_skip = frame_skip
        self._isTrain = isTrain

        self.bullet_setup(self._bullet_connection)
        bullet.setTimeStep(1./step_freq, self.clientID)

        self.goal = GoalMarker(self.clientID) # for visualizing the direction of transport.
        self.plane = Plane(self.clientID) # support surface representing ground
        self.cone = TrainObject(self.clientID) # object to be transported: here, a moai model

        self.cone.generate_urdf_file()

        action_low = np.array([-1, -1], dtype=np.float64)
        action_high = np.array([1, 1], dtype=np.float64)
        self.action_space = gym.spaces.box.Box(low=action_low, high=action_high)

        obs_low = np.array([-5, -5, -5, -10, -10, -10], dtype=np.float64)
        obs_high = np.array([5, 5, 5, 10, 10, 10], dtype=np.float64)
        self.observation_space = gym.spaces.box.Box(low=obs_low, high=obs_high)

        self.np_random, _ = gym.utils.seeding.np_random()
        self.reset()

    def step(self, action):

        duration = time.time()-self.start_time
        if duration > EPISODE_TIMEOUT:
            # print("terminated: timout")
            self.done = True

        if self._isTrain==True:
            action_scale = 0.25
        else:
            action_scale = 0.1

        self.cone.apply_action(action*action_scale)

        for _ in range(self._frame_skip):
            bullet.stepSimulation()

        true_cone_state, true_cone_te = self.cone.get_observation()
        noisy_cone_state = self.cone.get_noisy_observation(self.np_random)

        reward = self.set_rewards(true_cone_state, true_cone_te, action*action_scale)

        if self._bullet_connection == 2:
            self.adjust_camera_pose()

        ob = np.array([noisy_cone_state[2], noisy_cone_state[3], noisy_cone_state[4],
                       noisy_cone_state[7], noisy_cone_state[8], noisy_cone_state[9]], dtype=np.float64)

        return ob, reward, self.done, dict()


    def reset(self):
        self.done = False
        bullet.resetSimulation(self.clientID)
        bullet.setGravity(0, 0, -9.8)

        yaw_spawn = np.pi/2
        mu_cone_ground = np.inf
        self.initialize_physical_objects(yaw_spawn, mu_cone_ground)

        self.start_time = time.time()
        if self._bullet_connection == 2:
            self.adjust_camera_pose()

        true_cone_state, true_cone_te = self.cone.get_observation()
        noisy_cone_state = self.cone.get_noisy_observation(self.np_random)

        self.prev_x = [true_cone_state[0]]
        self.prev_a = [0,0]
        self.action_mag_sum = 0
        self._prev_time = time.time()

        ob = np.array([noisy_cone_state[2], noisy_cone_state[3], noisy_cone_state[4],
                       noisy_cone_state[7], noisy_cone_state[8], noisy_cone_state[9]], dtype=np.float64)

        return ob


    def set_rewards(self, cone_state, cone_te, action):

        action_accel = np.linalg.norm(action-np.array(self.prev_a))
        action_mag = np.linalg.norm(action)

        if cone_state[3]<np.radians(5) or cone_state[3]>np.radians(25):
            print("terminated: nutation out of bound")
            self.done = True
            reward = -50

        else:
            r_forward = 1000*(cone_state[0]-self.prev_x[0])
            r_spin = -20*max(abs(cone_state[4])-np.radians(20), 0) #20
            r_action_dot = -3*action_accel

            reward = r_forward + r_spin + r_action_dot

        self.prev_x = [cone_state[0]]
        self.prev_a = [action[0], action[1]]


        return reward

    def initialize_physical_objects(self,yaw_spawn,mu_cone_ground):
        self.goal.load_model_from_urdf()

        self.plane.load_model_from_urdf()
        self.plane.load_texutre()
        self.plane.set_lateral_friction(mu_cone_ground)
        self._planeID = self.plane.get_ids()[0]

        self.cone.load_model_from_urdf(yaw_spawn)
        self.cone.load_texutre()
        self.cone.set_lateral_friction(mu_cone_ground)
        self._coneID = self.cone.get_ids()[0]


        if self._isTrain==True:
            self.initial_cone_tilting(theta_des=DESIRED_NUTATION)
            pass
        else:
            self.initial_cone_tilting(theta_des=DESIRED_NUTATION)
            pass


    def initial_cone_tilting(self, theta_des):
        theta = 0
        self.cone.apply_action([0.5,0])
        while theta < theta_des:
            bullet.stepSimulation()
            cone_state = self.cone.get_observation()[0]
            theta = cone_state[3]
        self.cone.apply_action([0.,0.])
        bullet.stepSimulation()


    def bullet_setup(self, bullet_connection):

        if bullet_connection == 0:
            self.clientID = bullet.connect(bullet.DIRECT)
            # bullet.setDefaultContactERP(0.9)
        elif bullet_connection == 1:
            self.clientID = bullet.connect(bullet.GUI)
        elif bullet_connection == 2:
            self.clientID = bullet.connect(bullet.SHARED_MEMORY)
            bullet.configureDebugVisualizer(bullet.COV_ENABLE_GUI,0)
            self._cam_dist = 2.5 #3
            self._cam_yaw = -90 -30  #-20
            self._cam_pitch =-5
        bullet.setPhysicsEngineParameter(enableFileCaching=0)


    def adjust_camera_pose(self):
        cone_pos_world = bullet.getLinkState(self._coneID,linkIndex=5,physicsClientId=self.clientID)[0]
        bullet.resetDebugVisualizerCamera(cameraDistance=self._cam_dist,
                                          cameraYaw=self._cam_yaw,
                                          cameraPitch=self._cam_pitch,
                                          cameraTargetPosition=[cone_pos_world[0], cone_pos_world[1], cone_pos_world[2]+0.5])

    def render(self):
        pass

    def close(self):
        bullet.disconnect(self.clientID)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]



# print("r_forward:", r_forward)
# print("r_spin:", r_spin)
# print("r_nutation:", r_nutation)
# print("r_action_dot:", r_action_dot)
# print("------------------------------")
