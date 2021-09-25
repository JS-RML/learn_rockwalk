import os
import urx
import copy
import rospy
import time
import math
import numpy as np
import gym
import rock_walk

from robot_utils import RobotSetup
from helper_functions import *
from stable_baselines3 import SAC

from geometry_msgs.msg import Vector3, Pose, Twist, TwistStamped

from subscribers import SubscriberRockwalk

KONG_IP = '192.168.1.10'


def load_model(filename):
    model = SAC.load(filename, device="cpu")
    return model

def shutdown():
    print("ROS node shutting down")


def get_observation(rockwalk_sub, object_param):

    obs = np.array([rockwalk_sub._body_euler.x,
                    rockwalk_sub._body_euler.y,
                    rockwalk_sub._body_euler.z,
                    rockwalk_sub._body_twist.twist.angular.x,
                    rockwalk_sub._body_twist.twist.angular.y,
                    rockwalk_sub._body_twist.twist.angular.z]+object_param)
    return obs

def transform_action_to_world_frame(obs, action_b):

    rot_sb = get_rot_transform_s_b(obs)
    vec_GC_s = get_vector_GC_s(obs, rot_sb)
    rot_mat = compute_tangent_plane_transform_at_C(obs, vec_GC_s)
    action_s = np.matmul(rot_mat,np.array([action_b[0], action_b[1], 0]))

    return action_s


def main():
    """
    Object shape: [ellipse_a, ellipse_b, apex_x, apex_y, apex_z]
    """
    ellipse_params=[0.35,0.35]
    apex_coordinates=[0,-0.35,1.3]
    object_param = ellipse_params + apex_coordinates

    freq = 10.0

    trained_model = load_model("./saved_policy/rw_model_3000000_steps")

    rospy.loginfo("Publishing rl actions")
    rl_action_pub = rospy.Publisher('rl_action', Vector3, queue_size=10)
    rockwalk_sub = SubscriberRockwalk()

    kong_setup = RobotSetup(KONG_IP)
    kong_setup.set_home_configuration()
    kong_robot = kong_setup.get_robot()
    rospy.sleep(3)

    real_begin = input("Press enter to execute REAL robot motion")
    if real_begin == "":

        rate = rospy.Rate(freq)
        dt = 1/freq
        prev_speed = 0.
        action_scale = 0.1 # for faster or slower end-effector motion

        while not rospy.is_shutdown():
            t_start = time.time()

            obs = get_observation(rockwalk_sub, object_param)
            action_b, _states = trained_model.predict(obs[:6], deterministic=True)

            action = transform_action_to_world_frame(obs, action_b)

            action_real = action*action_scale

            speed = np.linalg.norm(action_real)
            acc = 2.5*(speed-prev_speed)/dt
            kong_robot.speedl((action_real[0], action_real[1], 0, 0, 0, 0), acc, min_time=0.5)


            rate.sleep()
            dt = time.time()-t_start
            prev_speed = speed
            rl_action_pub.publish(Vector3(action_real[0], action_real[1], 0))

    else:
        rospy.on_shutdown(shutdown)

if __name__ == "__main__":
    rospy.init_node("rl_rockwalk_ur10", anonymous=True)
    main()
