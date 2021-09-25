import rospy
import copy
from geometry_msgs.msg import Vector3, Pose, Twist, TwistStamped, WrenchStamped
from std_msgs.msg import Float64



class SubscriberRockwalk:
    """
    Subscribed data used for feedback control of rockwalk.
    """

    def __init__(self):

        self._body_euler_sub = rospy.Subscriber("body_euler", Vector3, self.store_body_euler_data)
        self._body_twist_sub = rospy.Subscriber("body_twist", TwistStamped, self.store_body_twist_data)
        self._ground_contact_sub = rospy.Subscriber("ground_contact", Vector3, self.store_ground_contact_data)
        self._apex_position_sub = rospy.Subscriber("apex_coordinates", Vector3, self.store_apex_position_data)

        self._min_nutation_sub = rospy.Subscriber("min_nutation", Float64, self.store_min_nutation_data)

        self._pke_sub = rospy.Subscriber("pke", Float64, self.store_pke_data)
        self._dke_sub = rospy.Subscriber("dke", Float64, self.store_dke_data)
        self._dkedt_sub = rospy.Subscriber("dkedt", Float64, self.store_dkedt_data)

        self._average_forward_speed_sub = rospy.Subscriber("average_forward_speed", Float64, self.store_average_forward_speed_data)
        self._average_forward_speed_derivative_sub = rospy.Subscriber("average_forward_speed_derivative", Float64, self.store_average_forward_speed_derivative_data)

        self._cmd_vel_sub = rospy.Subscriber("cmd_vel", Twist, self.store_cmd_vel_data)

        self._kong_tcp_pose_sub = rospy.Subscriber("kong_tcp_pose", Pose, self.store_kong_tcp_pose_data)
        self._ft_wrench_sub = rospy.Subscriber("robotiq_ft_wrench", WrenchStamped, self.store_robotiq_wrench_data)

        rospy.sleep(3)


    def store_body_euler_data(self, body_euler_data):
        self._body_euler = copy.deepcopy(body_euler_data)

    def store_body_twist_data(self, body_twist_data):
        self._body_twist = copy.deepcopy(body_twist_data)

    def store_ground_contact_data(self, ground_contact_data):
        self._ground_contact = copy.deepcopy(ground_contact_data)

    def store_apex_position_data(self, apex_position_data):
        self._apex_position = copy.deepcopy(apex_position_data)

    def store_min_nutation_data(self, min_nutation_data):
        self._min_nutation = copy.deepcopy(min_nutation_data)

    def store_pke_data(self, pke_data):
        self._pke = copy.deepcopy(pke_data)

    def store_dke_data(self, dke_data):
        self._dke = copy.deepcopy(dke_data)

    def store_dkedt_data(self, dkedt_data):
        self._dkedt = copy.deepcopy(dkedt_data)

    def store_average_forward_speed_data(self, average_forward_speed_data):
        self._average_forward_speed = copy.deepcopy(average_forward_speed_data)

    def store_average_forward_speed_derivative_data(self, average_forward_speed_derivative_data):
        self._average_forward_speed_derivative = copy.deepcopy(average_forward_speed_derivative_data)

    def store_cmd_vel_data(self, cmd_vel_data):
        self._cmd_vel = copy.deepcopy(cmd_vel_data)

    def store_kong_tcp_pose_data(self, kong_tcp_pose_data):
        self._kong_tcp_pose = copy.deepcopy(kong_tcp_pose_data)

    def store_robotiq_wrench_data(self, robotiq_wrench_data):
        self._robotiq_wrench = copy.deepcopy(robotiq_wrench_data)
