import os
import gym
import numpy as np
import rock_walk
import time
import pybullet as bullet

from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from custom_callback import GenerateObjectCallback


class RLModel:

    def __init__(self, connection, freq, frame_skip, train):

        if train == True:
            radius=[0.35]
            apex_coordinates=[-0.35,1.5]
            object_param = radius + apex_coordinates

            with open("./Rock-Walk/rock_walk/envs/training_objects_params.txt", 'w') as f:
                f.write("radius,apex_y,apex_z\n")
                f.write(str(0)+","+str(0)+","+str(0)+"\n")
                f.write(str(object_param[0])+","+str(object_param[1])+","+str(object_param[2])+"\n")

            self._env = gym.make("RockWalk-v0",bullet_connection=connection,step_freq=freq,frame_skip=frame_skip,isTrain=train)
            self._env = Monitor(self._env, "./log")

        else:
            self._env = gym.make("RockWalk-v0",bullet_connection=connection,step_freq=freq,frame_skip=frame_skip,isTrain=train)


    def train_model(self):
        n_actions = self._env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=0*np.ones(n_actions), sigma=0.5*np.ones(n_actions)) #5 prev
        # self._model.set_random_seed(seed=999)

        self._model = SAC("MlpPolicy", self._env,
                          action_noise=action_noise,
                          batch_size=128,
                          train_freq= 64,
                          gradient_steps= 64,
                          learning_starts=20000,
                          verbose=1,
                          tensorboard_log = "./rockwalk_tb/")


        object_callback = GenerateObjectCallback(check_freq=20000)

        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./save/', name_prefix='rw_model')

        callback_list = CallbackList([object_callback,checkpoint_callback])

        self._model.learn(total_timesteps=5000000, log_interval=10, callback=callback_list)
        self._model.save_replay_buffer('./save/buffer')
        self._env.close()


    def test_model(self, freq):
        self._trained_model = SAC.load("./save/rw_model_3000000_steps", device="cpu")
        print("Trained model loaded")
        obs = self._env.reset()

        with open("./log/data.txt", "w") as f:
            f.write("time[1],cone_state[10],contact_coordinates[3],position_control_point[3],velocity_control_point[3]\n")

        for count in range(15000):
            action, _states = self._trained_model.predict(obs, deterministic=True)
            obs, rewards, dones, info = self._env.step(action)

            contact_info = bullet.getContactPoints(self._env._coneID, self._env._planeID, 5, -1)[0]
            contact_coordinates = contact_info[6]

            true_cone_state, true_cone_te = self._env.cone.get_observation()

            # action_s = self._env.cone.transform_action_to_world_frame(action)

            position_C = self._env.cone.get_control_point_position()
            velocity_C = self._env.cone.get_control_point_velocity()

            data = np.concatenate((np.array([time.time()]),
                                   np.array(true_cone_state),
                                   np.array(contact_coordinates),
                                   np.array(position_C),
                                   np.array(velocity_C)))

            with open("./log/data.txt", "a") as f:
                np.savetxt(f, [data], delimiter=',')

            time.sleep(1./240.)


def main():
    freq = 50
    frame_skip = 5
    train_begin = input("Type 'yes' to TRAIN model")
    if train_begin == "yes":
        rl_model = RLModel(0, freq, frame_skip, train=True)
        rl_model.train_model()

    test_begin = input("Type 'yes' to TEST model")
    if test_begin == "yes":
        freq = 240
        frame_skip = 1
        rl_model = RLModel(1, freq, frame_skip, train=False) #0: DIRECT 1: GUI 2: SHARED_MEMORY
        rl_model.test_model(freq)
    else:
        rl_model._env.close()


if __name__ == "__main__":
    main()
