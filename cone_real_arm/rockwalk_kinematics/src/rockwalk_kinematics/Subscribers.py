import rospy
import copy
from geometry_msgs.msg import PoseArray, PoseStamped, Quaternion, WrenchStamped, Vector3, Pose, TwistStamped

from tf import transformations as tfms


class SubscriberMotionShield:
    """Subscriber to QuaternionStamped and TwistStamped topics
       from 9-axis arduino motion shield
    """

    def __init__(self):

        rospy.loginfo("Subscribing to Quaternion Topic from Motion Shield")
        self._quaternion_sub =rospy.Subscriber("quat_motion_shield", Quaternion, self.store_quaternion_data)

        rospy.loginfo("Subscribing to Twist Topic from Motion Shield")
        self._twist_sub =rospy.Subscriber("twist_motion_shield", Vector3, self.store_twist_data)

        rospy.sleep(3)

    def store_quaternion_data(self, quaternion_data):

        self._imu_quaternion = tfms.unit_vector([quaternion_data.x/1000.0,
                                                 quaternion_data.y/1000.0,
                                                 quaternion_data.z/1000.0,
                                                 quaternion_data.w/1000.0])


    def store_twist_data(self, twist_data):
        self._imu_twist = copy.deepcopy(twist_data) #in deg/s



class SubscriberKinematicsNode:

    def __init__(self):
        rospy.loginfo("Subscribing to Object Pose Topic")
        self._object_pose_sub = rospy.Subscriber("object_pose", PoseStamped, self.store_object_pose_data)

        rospy.loginfo("Subscribing to Kong Pose Topic")
        self._kong_tcp_pose_sub = rospy.Subscriber("kong_tcp_pose", Pose, self.store_kong_tcp_pose_data)

        rospy.loginfo("Subscribing to Ground Contact Topic")
        self._ground_contact_sub = rospy.Subscriber("ground_contact", Vector3, self.store_ground_contact_data)

        rospy.loginfo("Subscribing to Apex Coordinates Topic")
        self._apex_coordinates_sub = rospy.Subscriber("apex_coordinates", Vector3, self.store_apex_coordinates_data)

        rospy.loginfo("Subscribing to Apex Velocity Topic")
        self._apex_velocity_sub = rospy.Subscriber("apex_velocity", Vector3, self.store_apex_velocity_data)

        rospy.loginfo("Subscribing to CoM Coordinates Topic")
        self._CoM_coordinates_sub = rospy.Subscriber("CoM_coordinates", Vector3, self.store_CoM_coordinates_data)

        rospy.loginfo("Subscribing to CoM Velocity Topic")
        self._CoM_velocity_sub = rospy.Subscriber("CoM_velocity", Vector3, self.store_CoM_velocity_data)

        rospy.loginfo("Subscribing to Next Waypoint Topic")
        self._next_waypoint_sub = rospy.Subscriber("next_waypoint", PoseArray, self.store_next_waypoint_data)

        rospy.loginfo("Subscribing to Angular Velocity Topic")
        self._angular_velocity_sub = rospy.Subscriber("angular_velocity", Vector3, self.store_angular_velocity_data)

        rospy.loginfo("Subscribing to Robotiq Wrench Topic")
        self._robotiq_wrench_sub = rospy.Subscriber("robotiq_ft_wrench", WrenchStamped, self.store_robotiq_wrench_data)

        rospy.loginfo("Subscribing to Body Euler Topic")
        self._body_euler_sub = rospy.Subscriber("body_euler", Vector3, self.store_body_euler_data)

        rospy.loginfo("Subscribing to Body Twist Topic")
        self._body_twist_sub = rospy.Subscriber("body_twist", TwistStamped, self.store_body_twist_data)


        rospy.sleep(3)

    def store_object_pose_data(self, pose_data):
        self._object_pose = copy.deepcopy(pose_data)

    def store_kong_tcp_pose_data(self, pose_data):
        self._kong_tcp_pose = copy.deepcopy(pose_data)

    def store_ground_contact_data(self, ground_contact_data):
        self._ground_contact = copy.deepcopy(ground_contact_data)

    def store_apex_coordinates_data(self, apex_coordinates_data):
        self._apex_coordinates = copy.deepcopy(apex_coordinates_data)

    def store_apex_velocity_data(self, apex_velocity_data):
        self._apex_velocity = copy.deepcopy(apex_velocity_data)

    def store_CoM_coordinates_data(self, CoM_coordinates_data):
        self._CoM_coordinates = copy.deepcopy(CoM_coordinates_data)

    def store_CoM_velocity_data(self, CoM_velocity_data):
        self._CoM_velocity = copy.deepcopy(CoM_velocity_data)

    def store_next_waypoint_data(self, next_waypoint_data):
        self._next_waypoint = copy.deepcopy(next_waypoint_data)

    def store_angular_velocity_data(self, angular_velocity_data):
        self._angular_velocity = copy.deepcopy(angular_velocity_data)

    def store_robotiq_wrench_data(self, robotiq_wrench_data):
        self._robotiq_wrench = copy.deepcopy(robotiq_wrench_data)

    def store_body_euler_data(self, body_euler_data):
        self._body_euler = copy.deepcopy(body_euler_data)

    def store_body_twist_data(self, body_twist_data):
        self._body_twist = copy.deepcopy(body_twist_data)
